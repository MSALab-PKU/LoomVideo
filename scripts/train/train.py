"""
LoomVideo training script.
"""

import argparse
import contextlib
import gc
import json
import logging
import os
import sys
import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["NCCL_TIMEOUT"] = "3600"
os.environ["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] = "3600"
sys.path.append(os.getcwd())

from contextlib import nullcontext
from datetime import datetime, timedelta

import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
from omegaconf import OmegaConf
from torch.serialization import add_safe_globals
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from deepspeed.runtime.fp16.loss_scaler import LossScaler
from deepspeed.runtime.zero.config import ZeroStageEnum
from deepspeed.utils.tensor_fragment import fragment_address

# Extend NCCL group creation timeout to avoid transient timeouts
_original_new_group = c10d.new_group


def _new_group_with_long_timeout(*args, **kwargs):
    kwargs["timeout"] = timedelta(seconds=3600)
    return _original_new_group(*args, **kwargs)


c10d.new_group = _new_group_with_long_timeout
dist.new_group = _new_group_with_long_timeout

add_safe_globals([LossScaler, ZeroStageEnum, fragment_address])

from accelerate import Accelerator, DeepSpeedPlugin, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, set_seed

import diffusers
import transformers
from diffusers.training_utils import EMAModel
from transformers import optimization

from src.dataset.dataloader import (
    InfiniteTokenBatchedDataset,
    collate_fn,
    send_to_device,
)
from src.dataset.dataset import UniTrainDataset
from src.evaluation.generation import generate_eval_outputs
from src.evaluation.metrics import calculate_metrics
from src.models.utils import load_checkpoint, load_model

logger = get_logger(__name__, log_level="INFO")
logging.getLogger("PIL").setLevel(logging.WARNING)


def val_collate_fn(batch):
    """Collate function for validation that filters out None samples."""
    filtered = [s for s in batch if s is not None]
    if len(filtered) == 0:
        return None
    return collate_fn(filtered)


def build_accelerator(config):
    """Build Accelerator with optional DeepSpeed plugin."""
    is_distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1

    if is_distributed:
        with open(config.deepspeed_config_path) as f:
            ds_cfg = json.load(f)

        if (
            "train_micro_batch_size_per_gpu" not in ds_cfg
            or ds_cfg["train_micro_batch_size_per_gpu"] == "auto"
        ):
            ds_cfg["train_micro_batch_size_per_gpu"] = 1

        ds_cfg.setdefault("bf16", {})["enabled"] = True
        ds_cfg.setdefault("fp16", {})["enabled"] = False
        ds_plugin = DeepSpeedPlugin(hf_ds_config=ds_cfg)
    else:
        logger.info("Single-process mode detected. DeepSpeed plugin disabled.")
        ds_plugin = None

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    process_group_kwargs = InitProcessGroupKwargs(
        timeout=timedelta(seconds=3600)
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision="bf16",
        deepspeed_plugin=ds_plugin,
        kwargs_handlers=[ddp_kwargs, process_group_kwargs],
    )
    return accelerator


@contextlib.contextmanager
def ema_swap(ema: EMAModel, parameters_list):
    """Temporarily copy EMA weights into model, then restore originals."""
    params = list(parameters_list)
    ema.store(params)
    ema.copy_to(params)
    try:
        yield
    finally:
        ema.restore(params)


def save_ckpt(
    log_dir: str,
    global_step: int,
    accelerator: Accelerator,
    model,
    ema_model,
    config,
):
    """Save a full training checkpoint (DeepSpeed state + inference weights + EMA)."""
    time_start = time.perf_counter()

    ckpt_dir = os.path.join(log_dir, f"step-{global_step}", "checkpoints")
    accel_dir = os.path.join(ckpt_dir, "accelerator")
    if accelerator.is_main_process:
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(accel_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Save DeepSpeed / Accelerator engine state (optimizer, scheduler, etc.)
    logger.info("Saving Accelerator state...")
    accelerator.save_state(accel_dir)
    accelerator.wait_for_everyone()
    time_accel = time.perf_counter()

    if accelerator.is_main_process:
        logger.info(
            f"[Timer] Accelerator state save: {time_accel - time_start:.2f}s"
        )

        # Save per-component inference weights
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.eval()
        components = unwrapped_model.get_trainable_components()
        latest_dir = os.path.join(ckpt_dir, "latest")
        os.makedirs(latest_dir, exist_ok=True)

        for name, component in components.items():
            time_comp = time.perf_counter()
            save_path = os.path.join(latest_dir, f"{name}.pth")
            state_dict = {k: v.cpu() for k, v in component.state_dict().items()}
            torch.save(state_dict, save_path)
            logger.info(
                f"Saved {name} weights to {save_path} "
                f"({time.perf_counter() - time_comp:.2f}s)"
            )

        # Save EMA weights
        if config.ema.use_ema and ema_model is not None:
            time_ema = time.perf_counter()
            ema_dir = os.path.join(ckpt_dir, "ema")
            os.makedirs(ema_dir, exist_ok=True)
            ema_state_dict = ema_model.state_dict()
            if "shadow_params" in ema_state_dict:
                ema_state_dict["shadow_params"] = [
                    p.cpu() for p in ema_state_dict["shadow_params"]
                ]
            torch.save(ema_state_dict, os.path.join(ema_dir, "ema.pth"))
            logger.info(
                f"Saved EMA weights ({time.perf_counter() - time_ema:.2f}s)"
            )

        # Save metadata for resume
        with open(os.path.join(accel_dir, "trainer_meta.json"), "w") as f:
            json.dump({"global_step": int(global_step)}, f)

    accelerator.wait_for_everyone()
    time_end = time.perf_counter()
    if accelerator.is_main_process:
        logger.info(
            f"[Timer] Total checkpoint save: {time_end - time_start:.2f}s"
        )
        logger.info(f"Checkpoint saved to {ckpt_dir}")


def main(args):
    if int(os.environ.get("WORLD_SIZE", 1)) > 1:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

    assert args.config_path is not None or args.resume_path is not None, (
        "Either --config_path or --resume_path is required"
    )

    # ---- Load config ----
    if args.resume_path:
        exp_root = os.path.dirname(args.resume_path.rstrip("/"))
        config = OmegaConf.load(os.path.join(exp_root, "config.yaml"))
        log_dir = exp_root
        config_folder, config_name, dt = args.resume_path.split("/")[-4:-1]
        tb_dir = os.path.join(
            config.log_dir, "tb", config_folder, config_name, dt
        )
    else:
        config = OmegaConf.load(args.config_path)
        args_dict = {k: v for k, v in vars(args).items() if v is not None}
        config = OmegaConf.merge(config, OmegaConf.create(args_dict))
        log_dir = None

    dataset_config = OmegaConf.load(config.dataset_config_path)

    # ---- Initialize accelerator ----
    accelerator = build_accelerator(config)
    if accelerator.is_main_process:
        print(
            f"[Main] Launched with {accelerator.num_processes} processes",
            flush=True,
        )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    logger.info(
        "\n"
        + "\n".join(
            f"{k}\t{v}"
            for k, v in OmegaConf.to_container(config, resolve=True).items()
        )
    )
    if accelerator.is_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # ---- Create log directory ----
    if args.resume_path:
        logger.info(f"Resuming run. Using existing log_dir: {log_dir}")
    else:
        config_name = args.config_path.split("/")[-1][:-5]
        config_folder = args.config_path.split("/")[-2]
        if accelerator.is_main_process:
            dt = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
            log_dir = os.path.join(
                config.log_dir, config_folder, config_name, dt
            )
            tb_dir = os.path.join(
                config.log_dir, "tb", config_folder, config_name, dt
            )
            os.makedirs(log_dir, exist_ok=True)
            os.makedirs(tb_dir, exist_ok=True)
            OmegaConf.save(config, os.path.join(log_dir, "config.yaml"))
        else:
            log_dir = None

    accelerator.wait_for_everyone()

    # Broadcast log_dir to all ranks
    if accelerator.num_processes > 1:
        obj = [log_dir]
        dist.broadcast_object_list(obj, src=0)
        log_dir = obj[0]

    accelerator.wait_for_everyone()
    set_seed(config.seed)

    # ---- Build model ----
    time_start = time.perf_counter()
    model = load_model(config)
    logger.info(f"[Init] Model built in {time.perf_counter() - time_start:.2f}s")

    # Load previous-stage checkpoint (if specified)
    if getattr(config.model, "pretrained_ckpt_path", None) is not None:
        time_load = time.perf_counter()
        logger.info(
            f"Loading previous-stage weights from "
            f"{config.model.pretrained_ckpt_path}..."
        )
        load_checkpoint(model, config.model.pretrained_ckpt_path)
        logger.info(
            f"[Init] Previous-stage weights loaded in "
            f"{time.perf_counter() - time_load:.2f}s"
        )

    # Log trainable parameters
    logger.info("Trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(f"  {name}: {param.numel()}")

    total_trainable = sum(
        p.numel() for p in model.get_trainable_parameters()
    )
    logger.info(f"Total trainable parameters: {total_trainable / 1e6:.2f}M")

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(
        model.get_trainable_parameters(),
        lr=config.optimizer.learning_rate,
        betas=(config.optimizer.adam_beta1, config.optimizer.adam_beta2),
        weight_decay=config.optimizer.adam_weight_decay,
        eps=config.optimizer.adam_epsilon,
    )

    # Fix for DeepSpeed CPU Offload forcing CPUAdam (missing bias_correction key)
    for group in optimizer.param_groups:
        group.setdefault("bias_correction", True)

    # ---- LR scheduler ----
    lr_scheduler = optimization.get_constant_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.lr_scheduler.warmup_steps,
        last_epoch=-1,
    )

    # ---- Resume state ----
    if args.resume_path:
        ckpt_dir = os.path.join(args.resume_path, "checkpoints")
        accel_dir = os.path.join(ckpt_dir, "accelerator")
        meta_path = os.path.join(accel_dir, "trainer_meta.json")
        with open(meta_path, "r") as f:
            meta = json.load(f)
        global_step = int(meta.get("global_step", 0))
        logger.info(f"Resuming from global_step={global_step}")
    else:
        global_step = 0

    batches_to_skip = global_step * config.gradient_accumulation_steps
    estimated_samples_per_batch = config.data.train.estimated_samples_per_batch
    skip_samples = int(batches_to_skip * estimated_samples_per_batch)

    # ---- Datasets and dataloaders ----
    train_dataset = InfiniteTokenBatchedDataset(
        UniTrainDataset(
            dataset_config=dataset_config.train,
            config=config,
            dropout=True,
        ),
        max_batch_num_attention_tokens=config.data.train.max_batch_num_attention_tokens,
        seed=config.seed,
        num_processes=accelerator.num_processes,
        process_index=accelerator.process_index,
        skip_samples=skip_samples if args.resume_path is not None else 0,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=None,
        num_workers=config.data.train.num_workers,
        collate_fn=collate_fn,
        pin_memory=False,
    )

    validate_dataset = UniTrainDataset(
        dataset_config=dataset_config.val,
        config=config,
    )
    evaluation_dataset = UniTrainDataset(
        dataset_config=dataset_config.eval,
        config=config,
    )

    is_distributed = accelerator.num_processes > 1
    val_sampler = (
        DistributedSampler(
            validate_dataset,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            shuffle=False,
        )
        if is_distributed
        else None
    )
    val_dataloader = DataLoader(
        validate_dataset,
        batch_size=config.data.get("val", {}).get("batch_size", 1),
        sampler=val_sampler,
        shuffle=False,
        num_workers=config.data.train.num_workers,
        collate_fn=val_collate_fn,
        pin_memory=False,
        drop_last=False,
    )

    # ---- Prepare with Accelerator ----
    time_start = time.perf_counter()
    model, optimizer, lr_scheduler = accelerator.prepare(
        model, optimizer, lr_scheduler
    )
    logger.info(
        f"[Accelerator] Prepared model, optimizer, scheduler in "
        f"{time.perf_counter() - time_start:.2f}s"
    )

    # ---- Initialize EMA ----
    ema_model = None
    if config.ema.use_ema:
        unwrapped_model = accelerator.unwrap_model(model)
        ema_model = EMAModel(
            parameters=unwrapped_model.get_trainable_parameters(),
            decay=config.ema.ema_decay,
            update_after_step=config.ema.ema_start_step,
        )
        ema_model.to(accelerator.device, dtype=torch.float32)
        logger.info("EMA model initialized (FP32)")

    # ---- Load resume state ----
    if args.resume_path:
        time_start = time.perf_counter()
        accelerator.load_state(accel_dir)
        logger.info(
            f"[Resume] Loaded engine state in "
            f"{time.perf_counter() - time_start:.2f}s"
        )
        for group in optimizer.param_groups:
            group.setdefault("bias_correction", True)

        if config.ema.use_ema:
            ema_path = os.path.join(ckpt_dir, "ema", "ema.pth")
            ema_model.load_state_dict(
                torch.load(ema_path, map_location="cpu")
            )
            ema_model.to(accelerator.device, dtype=torch.float32)
            logger.info("Loaded EMA weights from checkpoint")

    # ---- TensorBoard ----
    tb_writer = None
    if accelerator.is_main_process:
        accelerator.init_trackers("LoomVideo")
        tb_writer = SummaryWriter(log_dir=tb_dir)

    # ---- Training loop ----
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset.dataset)}")
    logger.info(
        f"  Batch attention tokens per device = "
        f"{config.data.train.max_batch_num_attention_tokens}"
    )
    logger.info(
        f"  Gradient accumulation steps = "
        f"{accelerator.gradient_accumulation_steps}"
    )
    logger.info(f"  Total optimization steps = {config.train_steps}")

    progress_bar = tqdm(
        range(0, config.train_steps),
        initial=global_step,
        desc="Step",
        disable=not accelerator.is_main_process,
    )

    accelerator.wait_for_everyone()

    # Optional: save/generate before training starts
    if not args.resume_path and config.save_before_train:
        time_start = time.perf_counter()
        save_ckpt(log_dir, global_step, accelerator, model, ema_model, config)
        logger.info(
            f"[Save] step {global_step} in "
            f"{time.perf_counter() - time_start:.2f}s"
        )

    if not args.resume_path and config.generate_before_train:
        output_dir = os.path.join(log_dir, f"step-{global_step}", "eval")
        raw_model = accelerator.unwrap_model(model)
        raw_model.eval()
        ema_context = (
            ema_swap(ema_model, raw_model.get_trainable_parameters())
            if config.ema.use_ema
            else nullcontext()
        )
        with torch.no_grad(), ema_context, torch.autocast("cuda", dtype=torch.bfloat16):
            generate_eval_outputs(
                model=raw_model,
                dataset=evaluation_dataset,
                config=config,
                output_dir=output_dir,
                accelerator=accelerator,
            )

    # Accumulators for gradient-accumulated loss averaging
    accum_und_loss = torch.zeros((), device=accelerator.device, dtype=torch.float32)
    accum_gen_loss = torch.zeros((), device=accelerator.device, dtype=torch.float32)
    accum_count = 0

    while global_step < config.train_steps:
        for batch in train_dataloader:
            batch = send_to_device(batch, accelerator.device)
            model.train()

            with accelerator.accumulate(model):
                with accelerator.autocast(), torch.autocast(
                    "cuda", dtype=torch.bfloat16
                ):
                    und_loss, gen_loss = model.forward_loss(batch)

                total_loss = (
                    config.und_loss_weight * und_loss
                    + config.gen_loss_weight * gen_loss
                )
                accelerator.backward(
                    total_loss / config.gradient_accumulation_steps
                )

                accum_und_loss += und_loss.detach().to(torch.float32)
                accum_gen_loss += gen_loss.detach().to(torch.float32)
                accum_count += 1

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(
                    model.parameters(),
                    max_norm=config.optimizer.max_grad_norm,
                )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # Compute cross-rank averaged losses
                local_avg_und = accum_und_loss / max(accum_count, 1)
                local_avg_gen = accum_gen_loss / max(accum_count, 1)
                local_losses = torch.stack([local_avg_und, local_avg_gen])
                avg_losses = accelerator.reduce(local_losses, reduction="mean")
                avg_und_loss = avg_losses[0].item()
                avg_gen_loss = avg_losses[1].item()

                accum_und_loss.zero_()
                accum_gen_loss.zero_()
                accum_count = 0

                # Log to TensorBoard
                if accelerator.is_main_process:
                    tb_writer.add_scalar(
                        "Train/Und Loss", avg_und_loss, global_step
                    )
                    tb_writer.add_scalar(
                        "Train/Gen Loss", avg_gen_loss, global_step
                    )
                    progress_bar.set_postfix(
                        **{
                            "Und Loss": f"{avg_und_loss:.4f}",
                            "Gen Loss": f"{avg_gen_loss:.4f}",
                        }
                    )

                progress_bar.update(1)
                global_step += 1

                # EMA update
                if (
                    config.ema.use_ema
                    and global_step % config.ema.ema_interval == 0
                ):
                    with torch.no_grad():
                        ema_model.step(
                            accelerator.unwrap_model(
                                model
                            ).get_trainable_parameters()
                        )

                accelerator.wait_for_everyone()

                # ---- Periodic checkpointing ----
                if global_step % config.checkpointing_interval == 0:
                    time_start = time.perf_counter()
                    save_ckpt(
                        log_dir, global_step, accelerator,
                        model, ema_model, config,
                    )
                    logger.info(
                        f"[Save] step {global_step} in "
                        f"{time.perf_counter() - time_start:.2f}s"
                    )

                # ---- Periodic validation ----
                if global_step % config.validation_interval == 0:
                    _run_validation(
                        global_step, accelerator, model, ema_model,
                        validate_dataset, val_dataloader, config, tb_writer,
                    )

                # ---- Periodic evaluation ----
                if global_step % config.evaluation_interval == 0:
                    _run_evaluation(
                        global_step, accelerator, model, ema_model,
                        evaluation_dataset, dataset_config,
                        log_dir, config, tb_writer,
                    )

                # Periodic memory cleanup
                if global_step % config.get("clean_interval", 500) == 0:
                    gc.collect()
                    torch.cuda.empty_cache()

                if global_step >= config.train_steps:
                    break

    accelerator.wait_for_everyone()
    accelerator.end_training()
    logger.info("***** Training complete *****")


def _run_validation(
    global_step, accelerator, model, ema_model,
    validate_dataset, val_dataloader, config, tb_writer,
):
    """Run validation loss computation across all validation datasets."""
    logger.info(f"[Validation] step {global_step} start")
    time_start = time.perf_counter()

    all_target_names = sorted(list(validate_dataset.dataset_names))
    name_to_idx = {name: i for i, name in enumerate(all_target_names)}
    num_categories = len(all_target_names)

    noise_buckets = ["total", "low_noise", "mid_noise", "high_noise"]
    local_losses_dict = {
        bucket: torch.zeros(
            num_categories, device=accelerator.device, dtype=torch.float32
        )
        for bucket in noise_buckets
    }
    local_counts_vec = torch.zeros(
        num_categories, device=accelerator.device, dtype=torch.int32
    )

    model.eval()
    raw_model = accelerator.unwrap_model(model)
    val_ema_context = (
        ema_swap(ema_model, raw_model.get_trainable_parameters())
        if config.ema.use_ema and ema_model is not None
        else nullcontext()
    )

    with val_ema_context:
        for val_batch in val_dataloader:
            if val_batch is None:
                continue
            val_batch = send_to_device(val_batch, accelerator.device)

            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                per_sample_results = raw_model.get_val_loss_batch(
                    val_batch, num_valloss_timesteps=20
                )

            for sample_idx, sample_result in enumerate(per_sample_results):
                dataset_name = val_batch["dataset_names"][sample_idx]
                target_idx = name_to_idx[dataset_name]
                for bucket in noise_buckets:
                    local_losses_dict[bucket][target_idx] += sample_result[
                        bucket
                    ]
                local_counts_vec[target_idx] += 1

    # Reduce across all ranks
    global_losses_dict = {
        bucket: accelerator.reduce(
            local_losses_dict[bucket], reduction="sum"
        )
        for bucket in noise_buckets
    }
    global_count_vec = accelerator.reduce(local_counts_vec, reduction="sum")

    if accelerator.is_main_process:
        for i, name in enumerate(all_target_names):
            total_count = global_count_vec[i].item()
            if total_count > 0:
                for bucket in noise_buckets:
                    avg_loss = (
                        global_losses_dict[bucket][i].item() / total_count
                    )
                    logger.info(
                        f"Validation Gen Loss ({name}/{bucket}): "
                        f"{avg_loss:.4f}"
                    )
                    tb_writer.add_scalar(
                        f"Validation/Gen Loss/{name}/{bucket}",
                        avg_loss,
                        global_step,
                    )
            else:
                logger.info(
                    f"Validation Gen Loss ({name}): N/A (no samples)"
                )

    logger.info(
        f"[Validation] step {global_step} in "
        f"{time.perf_counter() - time_start:.2f}s"
    )


def _run_evaluation(
    global_step, accelerator, model, ema_model,
    evaluation_dataset, dataset_config,
    log_dir, config, tb_writer,
):
    """Run generation + metric evaluation on evaluation datasets."""
    time_start = time.perf_counter()
    output_dir = os.path.join(log_dir, f"step-{global_step}", "eval")
    raw_model = accelerator.unwrap_model(model)
    raw_model.eval()

    ema_context = (
        ema_swap(ema_model, raw_model.get_trainable_parameters())
        if config.ema.use_ema
        else nullcontext()
    )

    with torch.no_grad(), ema_context, torch.autocast(
        "cuda", dtype=torch.bfloat16
    ):
        generate_eval_outputs(
            model=raw_model,
            dataset=evaluation_dataset,
            config=config,
            output_dir=output_dir,
            accelerator=accelerator,
        )
        calculate_metrics(
            config=config,
            dataset_config=dataset_config,
            eval_root=output_dir,
            accelerator=accelerator,
            logger=logger,
        )

    # Log metric scores to TensorBoard
    if accelerator.is_main_process:
        for dataset_name in dataset_config.eval:
            metrics_dir = os.path.join(output_dir, dataset_name, "metrics")
            if not os.path.isdir(metrics_dir):
                continue
            for filename in os.listdir(metrics_dir):
                if not filename.endswith(".json"):
                    continue
                with open(
                    os.path.join(metrics_dir, filename), "r", encoding="utf-8"
                ) as f:
                    scores = json.load(f)
                for score_key, score_value in scores.items():
                    if isinstance(score_value, (int, float)):
                        tb_writer.add_scalar(
                            f"Metrics/{dataset_name}/{score_key}",
                            score_value,
                            global_step,
                        )
                    elif isinstance(score_value, dict):
                        for sub_key, sub_value in score_value.items():
                            if isinstance(sub_value, (int, float)):
                                tb_writer.add_scalar(
                                    f"Metrics/{dataset_name}/"
                                    f"{score_key}/{sub_key}",
                                    sub_value,
                                    global_step,
                                )

    accelerator.wait_for_everyone()
    logger.info(
        f"[Evaluation] step {global_step} in "
        f"{time.perf_counter() - time_start:.2f}s"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoomVideo training script")
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Path to training config YAML",
    )
    parser.add_argument(
        "--resume_path",
        type=str,
        default=None,
        help=(
            "Path to a checkpoint directory to resume from, "
            "e.g. /path/to/step-12000/checkpoints"
        ),
    )
    args = parser.parse_args()
    main(args)

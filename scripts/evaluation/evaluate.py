"""
LoomVideo evaluation script.
"""

import os
import sys
import copy
import time
import logging
import argparse
from pathlib import Path
from datetime import timedelta

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["NCCL_TIMEOUT"] = "3600"
os.environ["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] = "3600"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
import transformers
import diffusers
from omegaconf import OmegaConf
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import set_seed
from accelerate.logging import get_logger

from src.models.transformers.loomvideo import LoomVideo
from src.models.utils import load_checkpoint
from src.dataset.dataset import UniTrainDataset
from src.evaluation.generation import generate_eval_outputs
from src.evaluation.metrics import calculate_metrics

# Extend NCCL group timeout
_original_new_group = c10d.new_group


def _new_group_with_long_timeout(*args, **kwargs):
    kwargs["timeout"] = timedelta(seconds=3600)
    return _original_new_group(*args, **kwargs)


c10d.new_group = _new_group_with_long_timeout
dist.new_group = _new_group_with_long_timeout

logger = get_logger(__name__, log_level="INFO")
logging.getLogger("PIL").setLevel(logging.WARNING)


def _ensure_dir(path: Path, retries: int = 5, delay: float = 0.2) -> None:
    """Create directory with retries for filesystem propagation delays."""
    for _ in range(retries):
        try:
            path.mkdir(parents=True, exist_ok=True)
            return
        except FileNotFoundError:
            time.sleep(delay)
        except FileExistsError:
            return
        except OSError:
            time.sleep(delay)
    try:
        path.mkdir(parents=True, exist_ok=True)
    except FileExistsError:
        pass


def parse_eval_config(generation_configs_path: str, base_generation_config):
    """
    Parse the evaluation YAML into generation runs and dataset config.

    The YAML has a reserved ``dataset`` key for eval dataset definitions.
    All other top-level keys are generation parameter groups (each run
    inherits from ``base_generation_config`` and applies overrides).

    Returns:
        generation_runs: list of (run_name, generation_config) tuples
        eval_dataset_config: OmegaConf dict of dataset definitions
    """
    base = OmegaConf.to_container(base_generation_config, resolve=True)
    full_config = OmegaConf.load(generation_configs_path)

    eval_dataset_config = None
    if "dataset" in full_config:
        eval_dataset_config = OmegaConf.create(full_config.pop("dataset"))

    runs = []
    for run_name, overrides in full_config.items():
        merged = copy.deepcopy(base)
        merged.update(OmegaConf.to_container(overrides, resolve=True))
        runs.append((str(run_name), OmegaConf.create(merged)))

    return runs, eval_dataset_config


def main(args):
    # Initialise accelerator
    process_group_kwargs = InitProcessGroupKwargs(
        timeout=timedelta(seconds=3600)
    )
    accelerator = Accelerator(kwargs_handlers=[process_group_kwargs])

    # Load model config
    config = OmegaConf.load(args.config)

    # Parse evaluation YAML: generation runs + dataset config
    generation_runs, eval_dataset_config = parse_eval_config(
        args.generation_configs, config.generation
    )
    if eval_dataset_config is None:
        raise ValueError(
            f"No 'dataset' key found in {args.generation_configs}. "
            f"Please define evaluation datasets under a 'dataset' key."
        )

    # Logging setup
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # Set seed
    eval_seed = args.seed if args.seed is not None else config.seed
    set_seed(eval_seed)

    # Load model
    if accelerator.is_main_process:
        print("Initializing LoomVideo model...")
    model = LoomVideo(config)
    load_checkpoint(model, args.checkpoint_dir)
    model.to(dtype=torch.bfloat16, device=accelerator.device)

    # Prepare evaluation dataset
    if accelerator.is_main_process:
        print("Preparing evaluation dataset...")
    eval_dataset = UniTrainDataset(
        dataset_config=eval_dataset_config,
        config=config,
    )

    # Run generation for each parameter group
    for run_idx, (run_name, gen_config) in enumerate(generation_runs):
        output_dir = os.path.join(args.output_dir, run_name)
        _ensure_dir(Path(output_dir))

        config.generation = gen_config

        if accelerator.is_main_process:
            print(f"\n{'=' * 60}")
            print(
                f"[Run {run_idx + 1}/{len(generation_runs)}] {run_name}"
            )
            print(
                f"Generation config: "
                f"{OmegaConf.to_container(gen_config, resolve=True)}"
            )
            print(f"Output dir: {output_dir}")
            print(f"{'=' * 60}")

        generate_eval_outputs(
            model=model,
            dataset=eval_dataset,
            config=config,
            output_dir=output_dir,
            accelerator=accelerator,
        )

        if accelerator.is_main_process:
            print(f"Finish generation for run {run_name}!")

        if args.calculate_metrics:
            calculate_metrics(
                config=config,
                dataset_config=OmegaConf.create({"eval": eval_dataset_config}),
                eval_root=output_dir,
                accelerator=accelerator,
                logger=logger,
            )

    if accelerator.is_main_process:
        print(f"\nAll {len(generation_runs)} run(s) complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LoomVideo Evaluation: generate outputs and compute metrics."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the model config YAML",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help=(
            "Local directory containing model checkpoints. "
            "Expected structure: <dir>/ema/ema.pth or <dir>/latest/<component>.pth"
        ),
    )
    parser.add_argument(
        "--generation_configs",
        type=str,
        required=True,
        help=(
            "Path to evaluation YAML with generation param groups and "
            "a 'dataset' key defining eval datasets"
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/evaluation",
        help="Root directory for evaluation outputs",
    )
    parser.add_argument(
        "--calculate_metrics",
        action="store_true",
        default=False,
        help="If set, calculate metrics after generation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config.seed if provided)",
    )

    parsed_args = parser.parse_args()
    main(parsed_args)

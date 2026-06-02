"""
Generation output pipeline for evaluation.
"""

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from diffusers.utils.export_utils import export_to_video

from src.dataset.dataloader import send_to_device


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


def save_video(
    video_frames,
    fps: int,
    output_path: Path | str,
) -> None:
    """
    Save generated video frames or a single image to disk.

    Args:
        video_frames: numpy array of shape [T, H, W, C] with values in [0, 1].
        fps: Frames per second for video output.
        output_path: Target file path (extension will be adjusted for single frames).
    """
    output_path = Path(output_path)
    _ensure_dir(output_path.parent)

    is_single_frame = len(video_frames) == 1

    if is_single_frame:
        output_path = output_path.with_suffix(".jpg")
        frame = video_frames[0]
        frame = (frame * 255).clip(0, 255).astype(np.uint8)
        frame = Image.fromarray(frame)
        frame.save(str(output_path))
    else:
        export_to_video(video_frames, str(output_path), fps=fps)


def generate_eval_outputs(
    model,
    dataset,
    config,
    output_dir,
    accelerator,
):
    """
    Generate evaluation outputs for all samples in the dataset.

    Each rank processes its own shard of dataset indices. Generated outputs
    (images/videos) and their metadata JSONs are saved to ``output_dir``.

    Args:
        model: The LoomVideo model instance.
        dataset: UniTrainDataset with evaluation data.
        config: Full OmegaConf config.
        output_dir: Root output directory.
        accelerator: HuggingFace Accelerator instance.
    """
    output_dir = Path(output_dir)

    if accelerator.is_main_process:
        _ensure_dir(path=output_dir)
    accelerator.wait_for_everyone()

    if hasattr(model, "eval"):
        model.eval()

    # Split work across processes/ranks
    all_indices = list(range(len(dataset)))
    local_indices = all_indices[accelerator.process_index :: accelerator.num_processes]

    for idx in local_indices:
        data = dataset[idx]
        if data is None:
            continue

        dataset_name = data["dataset_name"]
        relative_path = data["gen_paths"][0]
        output_path = output_dir / dataset_name / "outputs" / relative_path

        if output_path.exists():
            print(f"[rank {accelerator.process_index}] Skipping existing output {output_path}")
            continue

        data = send_to_device(data, model.device)
        torch.cuda.empty_cache()

        # Determine generation shape from available pixel values
        if len(data["gen_pixel_values"]) != 0:
            gen_shape = list(data["gen_pixel_values"][0].shape[1:])
            if len(gen_shape) == 2:
                gen_shape = [1, *gen_shape]
        elif len(data["source_pixel_values"]) != 0:
            gen_shape = list(data["source_pixel_values"][0].shape[1:])
            if len(gen_shape) == 2:
                gen_shape = [1, *gen_shape]
        elif data["dataset_name"] == "GenEval":
            gen_shape = [1, 480, 832]
        elif "VBench" in data["dataset_name"]:
            gen_shape = [config.data.train.num_frames, 480, 832]
        else:
            # Default: use first ref image dimensions with configured frame count
            if len(data["ref_pixel_values"]) != 0:
                gen_shape = list(data["ref_pixel_values"][0].shape[1:])
                gen_shape = [config.data.train.num_frames, *gen_shape]
            else:
                gen_shape = [config.data.train.num_frames, 480, 832]

        generator = torch.Generator(device=accelerator.device).manual_seed(config.seed)

        # Use edit guidance scale when source conditioning is present
        guidance_scale = (
            config.generation.guidance_scale_edit
            if len(data["source_pixel_values"]) > 0
            else config.generation.guidance_scale
        )

        # Generate with retry for transient CUDA errors
        max_gen_retries = 3
        for gen_attempt in range(max_gen_retries):
            try:
                with accelerator.autocast():
                    output = model.generate(
                        **data["inputs"],
                        negative_prompt=config.generation.negative_prompt,
                        num_frames=gen_shape[0],
                        height=gen_shape[1],
                        width=gen_shape[2],
                        generator=generator,
                        guidance_scale=guidance_scale,
                        guidance_scale_visual=config.generation.guidance_scale_visual,
                        source_pixel_values=(
                            data["source_pixel_values"]
                            if len(data["source_pixel_values"]) > 0
                            else None
                        ),
                        ref_pixel_values=(
                            data["ref_pixel_values"]
                            if len(data["ref_pixel_values"]) > 0
                            else None
                        ),
                    )
                break
            except RuntimeError as e:
                if "cusolver" in str(e).lower() and gen_attempt < max_gen_retries - 1:
                    torch.cuda.empty_cache()
                    time.sleep(2.0)
                    generator = torch.Generator(device=accelerator.device).manual_seed(
                        config.seed
                    )
                    continue
                raise

        # Save generated output
        save_video(
            video_frames=output,
            output_path=output_path,
            fps=config.data.train.fps,
        )

        # Save companion data_info JSON
        json_path = output_path.with_suffix(".json")
        with open(json_path, "w") as f:
            json.dump(data["data_info"], f, indent=4, ensure_ascii=False)

    accelerator.wait_for_everyone()

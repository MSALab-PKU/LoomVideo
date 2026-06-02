"""
LoomVideo inference script.

Supports:
- t2v: Text-to-Video / Text-to-Image (num_frames=1)
- edit: Image/Video editing with source image or video + text instruction
- ref_edit: Image/Video editing with source + reference image(s) + text instruction
- mi2v: Multi-Image to Video generation from reference image(s) + text instruction
"""
import argparse
import os
import sys
import datetime
import logging
from typing import List, Optional

import numpy as np
import torch
from torchvision import transforms
from omegaconf import OmegaConf
from PIL import Image
from decord import VideoReader, cpu

from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers.utils.export_utils import export_to_video

from transformers import AutoProcessor

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

from src.models.utils import load_model, load_checkpoint
from src.models.transformers.loomvideo import LoomVideo

logger = logging.getLogger(__name__)

# Preprocessing: normalize to [-1, 1] for VAE/DiT conditioning
IMAGE_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

def resize_and_crop(image: Image.Image, target_height: int, target_width: int) -> Image.Image:
    """
    Resize and center-crop image to exact target dimensions.

    Args:
        image: PIL Image.
        target_height: Desired output height.
        target_width: Desired output width.

    Returns:
        Resized and cropped PIL Image.
    """
    width, height = image.size
    scale = max(target_width / width, target_height / height)
    image = transforms.functional.resize(
        image,
        (round(height * scale), round(width * scale)),
        interpolation=transforms.InterpolationMode.BILINEAR,
    )
    image = transforms.functional.center_crop(image, (target_height, target_width))
    return image


def load_image(image_path: str, target_height: int = None, target_width: int = None) -> tuple:
    """
    Load and preprocess a single image, returning both PIL and tensor.

    Args:
        image_path: Path to the image file.
        target_height: If set, resize image to this height.
        target_width: If set, resize image to this width.

    Returns:
        Tuple of (pil_image, tensor):
            pil_image: Resized PIL Image (for VLM input).
            tensor: Tensor of shape [C, H, W] normalized to [-1, 1] (for DiT conditioning).
    """
    image = Image.open(image_path).convert("RGB")
    if target_height is not None and target_width is not None:
        image = resize_and_crop(image, target_height, target_width)
    return image, IMAGE_TRANSFORM(image)

def load_images(
    image_paths: List[str], target_height: int = None, target_width: int = None
) -> tuple:
    """
    Load multiple images, returning both PIL images and tensors.

    Args:
        image_paths: List of image file paths.
        target_height: If set, resize each image to this height.
        target_width: If set, resize each image to this width.

    Returns:
        Tuple of (pil_images, tensors):
            pil_images: List of resized PIL Images.
            tensors: List of tensors, each [C, H, W] normalized to [-1, 1].
    """
    pil_images = []
    tensors = []
    for path in image_paths:
        pil_img, tensor = load_image(path, target_height, target_width)
        pil_images.append(pil_img)
        tensors.append(tensor)
    return pil_images, tensors


def load_video(
    video_path: str,
    target_fps: int = 24,
    target_num_frames: int = 121,
    scale_factor_temporal: int = 4,
    target_height: int = None,
    target_width: int = None,
) -> tuple:
    """
    Load and preprocess a video, returning both PIL frames and normalized tensor.

    Args:
        video_path: Path to the video file.
        target_fps: Target frame rate (must match training config).
        target_num_frames: Maximum number of frames to extract.
        scale_factor_temporal: VAE temporal scale factor for frame alignment.
        target_height: If set, resize each frame to this height.
        target_width: If set, resize each frame to this width.

    Returns:
        Tuple of (pil_frames, video_tensor):
            pil_frames: List of resized PIL Images (for VLM input).
            video_tensor: Tensor of shape [C, T, H, W] normalized to [-1, 1] (for DiT conditioning).
    """
    video_reader = VideoReader(video_path, ctx=cpu(0))
    video_fps = video_reader.get_avg_fps()
    num_video_frames = len(video_reader)

    # Compute target duration and in-range frames
    target_duration = (target_num_frames - 1) / target_fps
    num_video_in_range_frames = min(int(video_fps * target_duration) + 1, num_video_frames)

    # Start from the beginning of the video (no random crop at inference)
    start_frame_index = 0

    # Sample frames at target_fps rate
    frame_float = float(start_frame_index)
    interval = video_fps / target_fps

    frame_indices = []
    max_loops = target_num_frames * 3
    loop_count = 0

    while True:
        idx = min(round(frame_float), num_video_frames - 1)
        frame_indices.append(idx)
        frame_float += interval

        if (round(frame_float) - start_frame_index) >= num_video_in_range_frames:
            break

        loop_count += 1
        if loop_count > max_loops:
            break

    # Align to VAE temporal scale factor: (len - 1) // factor * factor + 1
    needed_len = (len(frame_indices) - 1) // scale_factor_temporal * scale_factor_temporal + 1
    frame_indices = frame_indices[:needed_len]

    # Decode and transform
    batch = video_reader.get_batch(frame_indices)
    video_data = batch.asnumpy()  # [T, H, W, C]

    pil_frames = []
    frame_tensors = []
    for frame in video_data:
        frame_pil = Image.fromarray(frame)
        if target_height is not None and target_width is not None:
            frame_pil = resize_and_crop(frame_pil, target_height, target_width)
        pil_frames.append(frame_pil)
        frame_tensors.append(IMAGE_TRANSFORM(frame_pil))

    # Stack: [T, C, H, W] -> [C, T, H, W]
    video_tensor = torch.stack(frame_tensors, dim=0).permute(1, 0, 2, 3)
    return pil_frames, video_tensor

def temporal_downsample_for_vlm(pil_frames: List[Image.Image], config) -> List[Image.Image]:
    """
    Downsample video frames for VLM input, matching evaluation pipeline.

    Args:
        pil_frames: Full list of PIL frames after resize.
        config: OmegaConf config object.

    Returns:
        Downsampled list of PIL frames for VLM input.
    """
    downsample_factor = OmegaConf.select(
        config, "data.und_temporal_downsample_factor", default=15
    )
    total_frames = len(pil_frames)
    if total_frames > downsample_factor:
        indices = np.linspace(
            0, total_frames - 1,
            max(total_frames // downsample_factor, 1),
            dtype=int,
        ).tolist()
    else:
        indices = list(range(total_frames))
    return [pil_frames[i] for i in indices]

def _get_negative_prompt(args, config) -> str:
    """Resolve negative prompt from args or config."""
    if args.negative_prompt is not None:
        return args.negative_prompt
    return OmegaConf.select(config, "generation.negative_prompt", default="")


def _get_guidance_scale(args, config, is_edit: bool) -> float:
    """
    Resolve guidance scale: use edit-specific scale for edit tasks.
    """
    if args.guidance_scale is not None:
        return args.guidance_scale
    if is_edit:
        return OmegaConf.select(config, "generation.guidance_scale_edit", default=2.5)
    return OmegaConf.select(config, "generation.guidance_scale", default=5.0)


def _get_guidance_scale_visual(args, config) -> float:
    """Resolve visual guidance scale from args or config."""
    if args.guidance_scale_visual is not None:
        return args.guidance_scale_visual
    return OmegaConf.select(config, "generation.guidance_scale_visual", default=1.5)


def _get_generator(args, device) -> Optional[torch.Generator]:
    """Create a random generator if seed is set."""
    if args.seed >= 0:
        return torch.Generator(device=device).manual_seed(args.seed)
    return None


def prepare_t2v_inputs(model: LoomVideo, config, args) -> dict:
    """Prepare inputs for text-to-video generation (no visual conditioning)."""
    processor = AutoProcessor.from_pretrained(
        config.model.und.pretrained_model_path, trust_remote_code=True
    )

    # T2V defaults: 480x832, 81 frames
    height = args.height if args.height is not None else 480
    width = args.width if args.width is not None else 832
    num_frames = args.num_frames if args.num_frames is not None else 81

    # Add task-specific instruction prefix (matching training data pipeline)
    is_image = (num_frames == 1)
    prefix = "Generate an image: " if is_image else "Generate a video: "

    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": f"{prefix}{args.prompt}"}],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "negative_prompt": _get_negative_prompt(args, config),
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": _get_guidance_scale(args, config, is_edit=False),
        "guidance_scale_visual": _get_guidance_scale_visual(args, config),
        "generator": _get_generator(args, model.device),
    }


def prepare_edit_inputs(model: LoomVideo, config, args) -> dict:
    """Prepare inputs for image/video editing (source + text instruction).

    Aligns with evaluation pipeline: passes resized PIL images/frames to Qwen processor
    (not file paths), and applies 15x temporal downsample for video VLM input.
    """
    processor = AutoProcessor.from_pretrained(
        config.model.und.pretrained_model_path, trust_remote_code=True
    )

    # Detect if source is an image or video based on extension
    source_path = args.source_video_path
    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    is_source_image = source_path.lower().endswith(image_extensions)

    if is_source_image:
        # Image editing: load, resize, pass PIL to processor (matching evaluation)
        source_image = Image.open(source_path).convert("RGB")
        native_width, native_height = source_image.size

        height = args.height if args.height is not None else native_height
        width = args.width if args.width is not None else native_width

        source_image = resize_and_crop(source_image, height, width)
        source_tensor = IMAGE_TRANSFORM(source_image).to(model.device)
        num_frames = 1

        # Pass resized PIL image to processor (not file path)
        content = [
            {"type": "image", "image": source_image},
            {"type": "text", "text": f"Edit this image: {args.prompt}"},
        ]
    else:
        # Video editing: load with fps resampling and resize
        probe_reader = VideoReader(source_path, ctx=cpu(0))
        first_frame = probe_reader[0].asnumpy()
        native_height, native_width = first_frame.shape[:2]
        del probe_reader

        height = args.height if args.height is not None else native_height
        width = args.width if args.width is not None else native_width

        pil_frames, source_tensor = load_video(
            source_path, target_height=height, target_width=width
        )
        source_tensor = source_tensor.to(model.device)
        num_source_frames = source_tensor.shape[1]
        num_frames = args.num_frames if args.num_frames is not None else num_source_frames

        # 15x temporal downsample for VLM input (matching evaluation pipeline)
        vlm_frames = temporal_downsample_for_vlm(pil_frames, config)

        # Pass downsampled PIL frames to processor (not file path)
        content = [
            {"type": "video", "video": vlm_frames},
            {"type": "text", "text": f"Edit this video: {args.prompt}"},
        ]

    messages = [{"role": "user", "content": content}]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "pixel_values": inputs.get("pixel_values"),
        "pixel_values_videos": inputs.get("pixel_values_videos"),
        "image_grid_thw": inputs.get("image_grid_thw"),
        "video_grid_thw": inputs.get("video_grid_thw"),
        "negative_prompt": _get_negative_prompt(args, config),
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": _get_guidance_scale(args, config, is_edit=True),
        "guidance_scale_visual": _get_guidance_scale_visual(args, config),
        "source_pixel_values": [source_tensor],
        "generator": _get_generator(args, model.device),
    }


def prepare_ref_edit_inputs(model: LoomVideo, config, args) -> dict:
    """Prepare inputs for reference-guided video editing."""
    processor = AutoProcessor.from_pretrained(
        config.model.und.pretrained_model_path, trust_remote_code=True
    )

    # Determine target resolution from source video or reference image
    if args.height is not None and args.width is not None:
        height, width = args.height, args.width
    elif args.source_video_path:
        probe_reader = VideoReader(args.source_video_path, ctx=cpu(0))
        first_frame = probe_reader[0].asnumpy()
        height, width = first_frame.shape[0], first_frame.shape[1]
        del probe_reader
    else:
        ref_image = Image.open(args.ref_image_paths[0]).convert("RGB")
        width, height = ref_image.size

    if args.height is not None:
        height = args.height
    if args.width is not None:
        width = args.width

    # Load source video: get both PIL frames and tensor
    source_pil_frames, source_video_tensor = load_video(
        args.source_video_path, target_height=height, target_width=width
    )
    source_video_tensor = source_video_tensor.to(model.device)
    num_source_frames = source_video_tensor.shape[1]
    num_frames = args.num_frames if args.num_frames is not None else num_source_frames

    # Load reference images: get both PIL and tensors
    ref_pil_images, ref_tensors = load_images(
        args.ref_image_paths, target_height=height, target_width=width
    )
    ref_image_tensors = [t.to(model.device) for t in ref_tensors]

    # 15x temporal downsample for VLM input (matching evaluation pipeline)
    vlm_frames = temporal_downsample_for_vlm(source_pil_frames, config)

    # Build Qwen input: pass resized PIL images/frames (not file paths)
    content = []
    for ref_pil in ref_pil_images:
        content.append({"type": "image", "image": ref_pil})
    content.append({"type": "video", "video": vlm_frames})
    content.append({"type": "text", "text": f"Edit this video with reference images: {args.prompt}"})

    messages = [{"role": "user", "content": content}]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "pixel_values": inputs.get("pixel_values"),
        "pixel_values_videos": inputs.get("pixel_values_videos"),
        "image_grid_thw": inputs.get("image_grid_thw"),
        "video_grid_thw": inputs.get("video_grid_thw"),
        "negative_prompt": _get_negative_prompt(args, config),
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": _get_guidance_scale(args, config, is_edit=True),
        "guidance_scale_visual": _get_guidance_scale_visual(args, config),
        "source_pixel_values": [source_video_tensor],
        "ref_pixel_values": ref_image_tensors,
        "generator": _get_generator(args, model.device),
    }


def prepare_mi2v_inputs(model: LoomVideo, config, args) -> dict:
    """Prepare inputs for multi-image to video generation (MI2V)."""
    processor = AutoProcessor.from_pretrained(
        config.model.und.pretrained_model_path, trust_remote_code=True
    )

    # Determine target resolution from reference images
    ref_image = Image.open(args.ref_image_paths[0]).convert("RGB")
    native_width, native_height = ref_image.size

    height = args.height if args.height is not None else native_height
    width = args.width if args.width is not None else native_width
    num_frames = args.num_frames if args.num_frames is not None else 81

    # Load reference images: get both PIL and tensors
    ref_pil_images, ref_tensors = load_images(
        args.ref_image_paths, target_height=height, target_width=width
    )
    ref_image_tensors = [t.to(model.device) for t in ref_tensors]

    # Build Qwen input: pass resized PIL images (not file paths)
    content = []
    for ref_pil in ref_pil_images:
        content.append({"type": "image", "image": ref_pil})
    content.append({"type": "text", "text": f"Generate a video with reference images: {args.prompt}"})

    messages = [{"role": "user", "content": content}]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "pixel_values": inputs.get("pixel_values"),
        "image_grid_thw": inputs.get("image_grid_thw"),
        "negative_prompt": _get_negative_prompt(args, config),
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": _get_guidance_scale(args, config, is_edit=False),
        "guidance_scale_visual": _get_guidance_scale_visual(args, config),
        "ref_pixel_values": ref_image_tensors,
        "generator": _get_generator(args, model.device),
    }


def save_output(output_np: np.ndarray, output_path: str, fps: int = 24):
    """
    Save generation output as image (T=1) or video (T>1).

    Args:
        output_np: Output array of shape [T, H, W, C] with values in [0, 1].
        output_path: Output file path. Extension is auto-corrected based on content.
        fps: Frames per second (only used for video).
    """
    num_frames = output_np.shape[0]

    # Single frame → save as image (matching evaluation: .jpg format)
    if num_frames == 1:
        if output_path.endswith(".mp4"):
            output_path = output_path.replace(".mp4", ".jpg")
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        frame = (output_np[0] * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(frame).save(output_path)
        logger.info(f"Saved image to: {output_path}")
        return

    # Multiple frames → save as video using export_to_video (matching evaluation)
    if not output_path.endswith(".mp4"):
        output_path = output_path.rsplit(".", 1)[0] + ".mp4"
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    export_to_video(output_np, output_path, fps=fps)

    logger.info(f"Saved video to: {output_path} ({num_frames} frames, {fps} fps)")

def generate_task(model: LoomVideo, config, args, accelerator: Accelerator):
    """
    Unified generation entry point for all tasks.

    Args:
        model: Loaded LoomVideo model.
        config: OmegaConf config object.
        args: Parsed arguments.
        accelerator: Accelerator instance.
    """
    task = args.task

    # Prepare task-specific inputs
    if task == "t2v":
        generate_kwargs = prepare_t2v_inputs(model, config, args)
    elif task == "edit":
        generate_kwargs = prepare_edit_inputs(model, config, args)
    elif task == "ref_edit":
        generate_kwargs = prepare_ref_edit_inputs(model, config, args)
    elif task == "mi2v":
        generate_kwargs = prepare_mi2v_inputs(model, config, args)
    else:
        raise NotImplementedError(f"Task '{task}' is not yet implemented.")

    # Log resolved parameters after preparation
    resolved_height = generate_kwargs["height"]
    resolved_width = generate_kwargs["width"]
    resolved_num_frames = generate_kwargs["num_frames"]

    if accelerator.is_main_process:
        task_display = {
            "t2v": "Text-to-Video",
            "edit": "Video Editing",
            "ref_edit": "Reference-guided Editing",
            "mi2v": "Multi-Image to Video",
        }
        logger.info(f"Task: {task_display.get(task, task)}")
        logger.info(f"Prompt: {args.prompt}")
        if args.source_video_path:
            logger.info(f"Source: {args.source_video_path}")
        if args.ref_image_paths:
            logger.info(f"Reference images: {args.ref_image_paths}")
        logger.info(f"Resolution: {resolved_width}x{resolved_height}, Frames: {resolved_num_frames}")
        logger.info(
            f"Steps: {args.num_inference_steps}, "
            f"CFG scale: {generate_kwargs['guidance_scale']}, "
            f"Visual CFG: {generate_kwargs['guidance_scale_visual']}"
        )
        logger.info("Starting generation...")

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        output = model.generate(**generate_kwargs)

    # Save output (only on main process)
    if accelerator.is_main_process:
        # fps: user-specified > default 24 (matching evaluation config.data.train.fps=24)
        fps = args.fps if args.fps is not None else 24

        output_path = args.output_path
        if output_path is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            ext = "png" if resolved_num_frames == 1 else "mp4"
            output_path = f"outputs/loomvideo_{task}_{timestamp}.{ext}"

        save_output(output, output_path, fps=fps)

    accelerator.wait_for_everyone()


def main():
    parser = argparse.ArgumentParser(
        description="LoomVideo Inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--config_path", type=str, required=True, help="Path to YAML config file.")
    parser.add_argument(
        "--ckpt_path", type=str, default=None,
        help="Path to trained checkpoint (directory or file). "
             "Directory: loads ema/ema.pth or latest/<component>.pth. "
             "File: loads .safetensors or .pt directly.",
    )

    parser.add_argument(
        "--task", type=str, default="t2v",
        choices=["t2v", "edit", "ref_edit", "mi2v"],
        help="Generation task type.",
    )

    parser.add_argument("--prompt", type=str, required=True, help="Text prompt / editing instruction.")
    parser.add_argument("--negative_prompt", type=str, default=None,
                        help="Negative prompt for CFG. If not set, uses value from config.")
    parser.add_argument("--height", type=int, default=None,
                        help="Output height. If not set: t2v uses 480; edit/ref_edit infers from source.")
    parser.add_argument("--width", type=int, default=None,
                        help="Output width. If not set: t2v uses 832; edit/ref_edit infers from source.")
    parser.add_argument("--num_frames", type=int, default=None,
                        help="Number of output frames. If not set: t2v uses 81; edit infers from source.")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of denoising steps.")
    parser.add_argument("--guidance_scale", type=float, default=None,
                        help="Text CFG scale. If not set: t2v uses 5.0; edit/ref_edit uses 2.5 from config.")
    parser.add_argument("--guidance_scale_visual", type=float, default=None,
                        help="Visual CFG scale. If not set, uses 1.5 from config.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed. Set to -1 for random.")
    parser.add_argument("--fps", type=int, default=None, help="Output video FPS. Auto-detected if not set.")

    parser.add_argument(
        "--source_video_path", type=str, default=None,
        help="Path to source video for edit/ref_edit tasks.",
    )
    parser.add_argument(
        "--ref_image_paths", type=str, nargs="+", default=None,
        help="Path(s) to reference image(s) for ref_edit task. Supports multiple images.",
    )

    parser.add_argument("--output_path", type=str, default=None, help="Output file path.")

    args = parser.parse_args()

    # Validate task-specific arguments
    if args.task == "edit" and not args.source_video_path:
        parser.error("--source_video_path is required for 'edit' task.")
    if args.task == "ref_edit":
        if not args.source_video_path:
            parser.error("--source_video_path is required for 'ref_edit' task.")
        if not args.ref_image_paths:
            parser.error("--ref_image_paths is required for 'ref_edit' task.")
    if args.task == "mi2v":
        if not args.ref_image_paths:
            parser.error("--ref_image_paths is required for 'mi2v' task.")

    # Initialize Accelerator
    accelerator = Accelerator(mixed_precision="bf16")

    # Setup logging: only main process outputs logs to avoid duplicate prints
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        level=logging.INFO if accelerator.is_main_process else logging.WARNING,
    )

    # Set seed
    if args.seed >= 0:
        set_seed(args.seed)

    # Load config
    if not os.path.exists(args.config_path):
        raise FileNotFoundError(f"Config file not found: {args.config_path}")
    config = OmegaConf.load(args.config_path)
    if accelerator.is_main_process:
        logger.info(f"Loaded config from: {args.config_path}")

    # Build model
    if accelerator.is_main_process:
        logger.info("Building LoomVideo model...")
        logger.info(f"  Understanding backbone: {config.model.und.pretrained_model_path}")
        logger.info(f"  Generation backbone: {config.model.gen.pretrained_model_path}")

    model = load_model(config)

    # Load checkpoint (EMA priority)
    if args.ckpt_path:
        load_checkpoint(model, args.ckpt_path)
    elif accelerator.is_main_process:
        logger.info("No checkpoint provided. Using pretrained backbone weights only.")

    # Move to device
    device = accelerator.device
    model.to(dtype=torch.bfloat16, device=device)
    model.eval()

    accelerator.wait_for_everyone()

    # Run generation
    generate_task(model, config, args, accelerator)

    if accelerator.is_main_process:
        logger.info("Done.")


if __name__ == "__main__":
    main()

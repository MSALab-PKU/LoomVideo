from typing import Optional, Tuple
import os
import gc
import warnings
import json
import random
import decord
from PIL import Image
import numpy as np
import imageio
from omegaconf import DictConfig, ListConfig, OmegaConf

from transformers import AutoTokenizer, AutoProcessor

import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms

from . import processors
from .utils import get_closest_resolution, IMAGE_RESOLUTION_BUCKETS, VIDEO_RESOLUTION_BUCKETS


class UniTrainDataset(Dataset):
    """
    Unified training dataset that supports various generation and editing tasks.

    Args:
        dataset_config: OmegaConf config defining one or more sub-datasets.
        config: Global training config.
        is_debug: If True, save intermediate media to disk for inspection.
        dropout: If True, apply text/all dropout for classifier-free guidance training.
    """

    def __init__(
        self,
        dataset_config: DictConfig | ListConfig,
        config: DictConfig | ListConfig,
        is_debug: bool = False,
        dropout: bool = False,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model.und.pretrained_model_path, trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(
            config.model.und.pretrained_model_path, trust_remote_code=True
        )
        self.dataset_config = dataset_config
        self.dataset_names, self.dataset_list = self._flatten_dataset_config(dataset_config)
        self.config = config
        self.is_debug = is_debug
        self.dropout = dropout
        if self.dropout:
            self.text_dropout_rate = config.data.train.text_dropout_rate
            self.all_dropout_rate = config.data.train.all_dropout_rate

        if self.is_debug:
            self.debug_output_dir = "outputs"
            os.makedirs(self.debug_output_dir, exist_ok=True)
            print(f"Debug mode enabled. Saving images/videos to {self.debug_output_dir}")

        self.image_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        self.im_start_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.assistant_id = self.tokenizer.convert_tokens_to_ids("assistant")

        self._build_index_offset()

        # Log flattened dataset info
        if len(self.dataset_names) > 0 and hasattr(self.dataset_list[0], "sample_weight"):
            print("=" * 60)
            print("Dataset sampling weights (after hierarchical flattening):")
            for name, info in zip(self.dataset_names, self.dataset_list):
                weight = getattr(info, "sample_weight", "N/A")
                num = getattr(info, "num_samples", "N/A")
                print(f"  {name:40s}  weight={weight:.2f}  num_samples={num}")
            print("=" * 60)

        self._internal_counter = 0

    @staticmethod
    def _flatten_dataset_config(dataset_config):
        """
        Flatten a hierarchical dataset config into flat lists.

        Supports two formats:
        1. **Hierarchical** (task group): an entry has ``task_weight`` and
           ``datasets`` keys. Each child dataset has a ``sample_weight``
           that represents its relative weight within the group. The
           final absolute weight is::

               task_weight * (sample_weight / sum_of_group_sample_weights)

        2. **Flat** (legacy): an entry directly contains dataset fields
           like ``process_func_name``, ``num_samples``, etc.

        Returns:
            dataset_names: List[str]
            dataset_list:  List[DictConfig] (each with a computed ``sample_weight``)
        """
        dataset_names = []
        dataset_list = []

        for entry_name, entry_value in dataset_config.items():
            is_task_group = (
                OmegaConf.is_dict(entry_value)
                and "task_weight" in entry_value
                and "datasets" in entry_value
            )

            if is_task_group:
                task_weight = float(entry_value.task_weight)
                child_datasets = entry_value.datasets

                group_weight_sum = sum(
                    float(child.sample_weight) for child in child_datasets.values()
                )

                for child_name, child_config in child_datasets.items():
                    relative_weight = float(child_config.sample_weight)
                    absolute_weight = task_weight * (relative_weight / group_weight_sum)

                    flat_config = OmegaConf.to_container(child_config, resolve=True)
                    flat_config["sample_weight"] = absolute_weight
                    flat_config = OmegaConf.create(flat_config)

                    dataset_names.append(f"{entry_name}/{child_name}")
                    dataset_list.append(flat_config)
            else:
                dataset_names.append(entry_name)
                dataset_list.append(entry_value)

        return dataset_names, dataset_list


    def _build_index_offset(self):
        """Build cumulative index offsets for multi-dataset indexing."""
        offset = [0]
        for info in self.dataset_list:
            offset.append(offset[-1] + int(info.num_samples))
        self.index_offset = offset

    def _get_data_index(self, idx):
        """Map a global index to (dataset_index, local_data_index)."""
        for i in range(1, len(self.index_offset)):
            if idx < self.index_offset[i]:
                return i - 1, idx - self.index_offset[i - 1]


    @staticmethod
    def _read_json(path: str) -> dict:
        """Read a JSON file from the local filesystem."""
        with open(path, "r") as f:
            return json.load(f)

    @staticmethod
    def _read_image(path: str) -> Image.Image:
        """Read an image from the local filesystem."""
        return Image.open(path).convert("RGB")

    def _read_video(self, path: str, clip_start_ratio: Optional[float] = None):
        """
        Read video frames from a local file, sampling at target FPS.

        Applies temporal sampling to match the training FPS and frame count.
        Frames are aligned to the VAE temporal scale factor.

        Args:
            path: Local path to the video file.
            clip_start_ratio: Where to start the clip (0.0-1.0). Random if None.

        Returns:
            List of PIL Image frames.
        """
        try:
            vr = decord.VideoReader(path, ctx=decord.cpu(0))
        except Exception as e:
            raise ValueError(f"Decord failed to load video {path}: {e}")

        target_fps = self.config.data.train.fps
        target_num_frames = self.config.data.train.num_frames
        target_duration = (target_num_frames - 1) / target_fps

        video_fps = vr.get_avg_fps()
        num_video_frames = len(vr)

        if num_video_frames <= 0 or video_fps <= 0:
            del vr
            raise ValueError(
                f"Invalid video metadata for {path}: frames={num_video_frames}, fps={video_fps}"
            )

        if video_fps < target_fps // 2:
            warnings.warn(
                f"{path}: Video fps {video_fps} is lower than target fps {target_fps} // 2. "
                "Will load duplicate frames."
            )

        num_video_in_range_frames = min(
            int(video_fps * target_duration) + 1, num_video_frames
        )

        # Determine clip start position
        max_start_idx = max(0, num_video_frames - num_video_in_range_frames)
        if max_start_idx == 0:
            start_frame_index = 0
        else:
            if clip_start_ratio is None:
                clip_start_ratio = random.random()
            start_frame_index = min(
                int(round(clip_start_ratio * max_start_idx)), max_start_idx
            )

        # Sample frame indices at target FPS
        frame_float = float(start_frame_index)
        interval = video_fps / target_fps

        frame_indices = []
        max_while_loops = target_num_frames * 3
        loop_count = 0

        while True:
            idx = min(round(frame_float), num_video_frames - 1)
            frame_indices.append(idx)
            frame_float = frame_float + interval

            if (round(frame_float) - start_frame_index) >= num_video_in_range_frames:
                break

            loop_count += 1
            if loop_count > max_while_loops:
                break

        # Align to VAE temporal scale factor: keep (k * scale_factor + 1) frames
        scale_factor_temporal = self.config.model.gen.vae.scale_factor_temporal
        needed_len = (
            (len(frame_indices) - 1) // scale_factor_temporal * scale_factor_temporal + 1
        )
        frame_indices = frame_indices[:needed_len]

        try:
            batch = vr.get_batch(frame_indices)
            video_data = (
                batch.numpy() if isinstance(batch, torch.Tensor) else batch.asnumpy()
            )
        except Exception as e:
            del vr
            raise ValueError(f"Decord get_batch failed on {path}: {e}")

        del vr

        frames = [Image.fromarray(v) for v in video_data]
        del video_data

        return frames


    def _resize_pixel_values_spatial(
        self, pixel_values: torch.Tensor, size: Tuple[int, int]
    ):
        """
        Resize pixel values to a target spatial size using bilinear interpolation.

        Args:
            pixel_values: 3D image tensor [C, H, W] or 4D video tensor [C, T, H, W].
            size: (target_height, target_width).
        """
        target_height, target_width = size
        if pixel_values.ndim == 3:
            return torchvision.transforms.functional.resize(
                pixel_values,
                (target_height, target_width),
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
            )

        if pixel_values.ndim == 4:
            frames_first = pixel_values.transpose(0, 1)  # [T, C, H, W]
            resized = torchvision.transforms.functional.resize(
                frames_first,
                (target_height, target_width),
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
            )
            return resized.transpose(0, 1)

        raise ValueError("pixel_values must be a 3D image tensor or 4D video tensor")

    @staticmethod
    def _get_aligned_spatial_size(
        src_height: int, src_width: int, gen_height: int, gen_width: int
    ):
        """Determine the aligned spatial size: use the smaller-pixel resolution."""
        src_pixels = src_height * src_width
        gen_pixels = gen_height * gen_width
        if src_pixels <= gen_pixels:
            return src_height, src_width
        return gen_height, gen_width

    def _resize_and_crop(self, image, res_buckets):
        """
        Resize and center-crop an image/frame to the closest resolution bucket.

        Uses scale-then-crop strategy: scale so the shorter side matches,
        then center-crop to exact bucket dimensions.
        """
        width, height = image.size
        target_height, target_width = get_closest_resolution(
            width, height, res_buckets, self.config.data.train.resolution_buckets
        )
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height * scale), round(width * scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
        )
        image = torchvision.transforms.functional.center_crop(
            image, (target_height, target_width)
        )
        return image
        

    def _get_gen_image_num_attention_tokens(self, image: torch.Tensor):
        """Compute the number of DiT attention tokens for a single image."""
        height, width = image.shape[-2:]
        h_latent = height // self.config.model.gen.vae.scale_factor_spatial
        w_latent = width // self.config.model.gen.vae.scale_factor_spatial
        h_patch = h_latent // self.config.model.gen.transformer.patch_size[1]
        w_patch = w_latent // self.config.model.gen.transformer.patch_size[2]
        return h_patch * w_patch

    def _get_gen_video_num_attention_tokens(self, video: torch.Tensor):
        """Compute the number of DiT attention tokens for a video."""
        num_frames, height, width = video.shape[-3:]
        t_latent = (num_frames - 1) // self.config.model.gen.vae.scale_factor_temporal + 1
        h_latent = height // self.config.model.gen.vae.scale_factor_spatial
        w_latent = width // self.config.model.gen.vae.scale_factor_spatial
        t_patch = t_latent // self.config.model.gen.transformer.patch_size[0]
        h_patch = h_latent // self.config.model.gen.transformer.patch_size[1]
        w_patch = w_latent // self.config.model.gen.transformer.patch_size[2]
        return t_patch * h_patch * w_patch

    # --- Debug Helpers ---
    def _denormalize(self, tensor):
        """Convert a normalized tensor back to uint8 numpy for visualization."""
        t = tensor.clone().detach().cpu()
        t = t * 0.5 + 0.5
        t = torch.clamp(t, 0, 1) * 255
        return t.byte()

    def _save_debug_media(self, media, name_suffix):
        """Save media (PIL Image, frame list, or tensor) to disk for debugging."""
        save_path_base = os.path.join(self.debug_output_dir, name_suffix)

        if isinstance(media, torch.Tensor):
            if media.ndim == 3:  # [C, H, W] -> image
                arr = self._denormalize(media).permute(1, 2, 0).numpy()
                Image.fromarray(arr).save(f"{save_path_base}.jpg")
            elif media.ndim == 4:  # [C, T, H, W] -> video
                arr = self._denormalize(media).permute(1, 2, 3, 0).numpy()
                imageio.mimsave(f"{save_path_base}.mp4", arr, fps=self.config.data.train.fps)

        elif isinstance(media, list) and isinstance(media[0], Image.Image):
            imageio.mimsave(f"{save_path_base}.mp4", media, fps=self.config.data.train.fps)

        elif isinstance(media, Image.Image):
            media.save(f"{save_path_base}.jpg")


    def __len__(self):
        return self.index_offset[-1]

    def _getitem(self, dataset_index, data_index):
        """
        Load and process a single data sample.

        Reads the JSON descriptor, applies the processor function to get segments,
        then assembles VLM inputs (user_content), DiT inputs (pixel values), and labels.
        """
        dataset_info = self.dataset_list[dataset_index]

        # Read sample metadata from local filesystem
        data_json_dir = dataset_info.data_json_dir
        json_path = os.path.join(data_json_dir, f"{data_index}.json")
        data_info = self._read_json(json_path)
        data_info["_data_index"] = data_index

        # Apply task-specific processor to get segment list
        processor_fn = getattr(processors, dataset_info.process_func_name)
        segments = processor_fn(dataset_info, data_info)

        # Initialize accumulators
        user_content = []
        assistant_content = []
        gen_pixel_values = []
        has_gen_output = False
        num_attention_tokens = 0
        num_gen_attention_tokens = 0
        prompts = []
        und_paths = []
        und_types = []
        gen_paths = []
        source_pixel_values = []
        ref_pixel_values = []

        # Pair source and target videos for consistent clip start positions
        num_source_videos = sum(
            1 for seg in segments if not seg["is_target"] and seg["type"] == "source_video"
        )
        num_target_videos = sum(
            1 for seg in segments if seg["is_target"] and seg["type"] == "video"
        )
        num_paired_videos = min(num_source_videos, num_target_videos)
        paired_video_clip_ratios = [random.random() for _ in range(num_paired_videos)]
        source_video_pair_idx = 0
        target_video_pair_idx = 0

        for seg_i, seg in enumerate(segments):
            debug_prefix = f"dataset{dataset_index}_data{data_index}_seg{seg_i}_{seg['type']}"

            if seg["is_target"] is False:
                if seg["type"] == "text":
                    user_content.append({"type": "text", "text": seg["content"]})

                    # For evaluation (no target image/video)
                    if "save_path" in seg:
                        gen_paths.append(seg["save_path"])
                        has_gen_output = True

                elif seg["type"] == "image":
                    image = self._read_image(seg["path"])
                    if self.is_debug:
                        self._save_debug_media(image, f"{debug_prefix}_orig")

                    if seg["need_resize"]:
                        image = self._resize_and_crop(image, IMAGE_RESOLUTION_BUCKETS)

                    if self.is_debug:
                        self._save_debug_media(image, f"{debug_prefix}_processed")

                    pixel_values = self.image_transform(image)

                    if self.is_debug:
                        self._save_debug_media(pixel_values, f"{debug_prefix}_transformed")

                    user_content.append({"type": "image", "image": image})
                    und_paths.append(seg["rel_path"])
                    und_types.append("ref")
                    ref_pixel_values.append(pixel_values)
                    tokens = self._get_gen_image_num_attention_tokens(pixel_values)
                    num_gen_attention_tokens += tokens

                elif seg["type"] == "video":
                    raise NotImplementedError(
                        "Reference video (is_target=False, type=video) is not supported yet."
                    )

                elif seg["type"] == "source_image":
                    image = self._read_image(seg["path"])
                    if self.is_debug:
                        self._save_debug_media(image, f"{debug_prefix}_orig")

                    if seg["need_resize"]:
                        image = self._resize_and_crop(image, IMAGE_RESOLUTION_BUCKETS)

                    if self.is_debug:
                        self._save_debug_media(image, f"{debug_prefix}_processed")

                    pixel_values = self.image_transform(image)

                    if self.is_debug:
                        self._save_debug_media(pixel_values, f"{debug_prefix}_transformed")

                    if self.config.model.und.add_source_video:
                        user_content.append({"type": "image", "image": image})
                    und_paths.append(seg["rel_path"])
                    und_types.append("source")
                    source_pixel_values.append(pixel_values)

                    if "save_path" in seg:
                        gen_paths.append(seg["save_path"])
                        has_gen_output = True

                elif seg["type"] == "source_video":
                    # Use paired clip ratio for consistent source/target alignment
                    clip_start_ratio = None
                    if source_video_pair_idx < num_paired_videos:
                        clip_start_ratio = paired_video_clip_ratios[source_video_pair_idx]
                    video_frames = self._read_video(seg["path"], clip_start_ratio=clip_start_ratio)
                    source_video_pair_idx += 1

                    if self.is_debug:
                        self._save_debug_media(video_frames, f"{debug_prefix}_orig")

                    if seg["need_resize"]:
                        video_frames = [
                            self._resize_and_crop(frame, VIDEO_RESOLUTION_BUCKETS)
                            for frame in video_frames
                        ]

                    if self.is_debug:
                        self._save_debug_media(video_frames, f"{debug_prefix}_processed")

                    # Full-resolution tensor for DiT conditioning
                    pixel_values = [self.image_transform(frame) for frame in video_frames]
                    pixel_values = torch.stack(pixel_values).transpose(0, 1)  # [C, T, H, W]

                    if self.is_debug:
                        self._save_debug_media(pixel_values, f"{debug_prefix}_transformed")

                    # Temporally downsampled version for VLM understanding
                    und_temporal_downsample_factor = self.config.data.train.und_temporal_downsample_factor
                    total_frames = len(video_frames)
                    if total_frames > und_temporal_downsample_factor:
                        und_frame_indices = np.linspace(
                            0,
                            total_frames - 1,
                            max(total_frames // und_temporal_downsample_factor, 1),
                            dtype=int,
                        ).tolist()
                    else:
                        und_frame_indices = list(range(total_frames))
                    und_video_frames = [video_frames[i] for i in und_frame_indices]

                    if self.config.model.und.add_source_video:
                        user_content.append({"type": "video", "video": und_video_frames})
                    und_paths.append(seg["rel_path"])
                    und_types.append("source")
                    source_pixel_values.append(pixel_values)

                    if "save_path" in seg:
                        gen_paths.append(seg["save_path"])
                        has_gen_output = True

                elif seg["type"] == "prompt":
                    prompts.append(seg["content"])

            elif seg["is_target"] is True:
                if seg["type"] == "text":
                    assistant_content.append({"type": "text", "text": seg["content"]})

                if seg["type"] == "image":
                    image = self._read_image(seg["path"])
                    if self.is_debug:
                        self._save_debug_media(image, f"{debug_prefix}_orig")

                    if seg["need_resize"]:
                        image = self._resize_and_crop(image, IMAGE_RESOLUTION_BUCKETS)

                    pixel_values = self.image_transform(image)

                    if self.is_debug:
                        self._save_debug_media(pixel_values, f"{debug_prefix}_transformed")

                    gen_pixel_values.append(pixel_values)
                    tokens = self._get_gen_image_num_attention_tokens(pixel_values)
                    num_gen_attention_tokens += tokens
                    num_attention_tokens += tokens
                    has_gen_output = True
                    gen_paths.append(seg["rel_path"])

                elif seg["type"] == "video":
                    # Use paired clip ratio for consistent source/target alignment
                    clip_start_ratio = None
                    if target_video_pair_idx < num_paired_videos:
                        clip_start_ratio = paired_video_clip_ratios[target_video_pair_idx]
                    video_frames = self._read_video(seg["path"], clip_start_ratio=clip_start_ratio)
                    target_video_pair_idx += 1

                    if self.is_debug:
                        self._save_debug_media(video_frames, f"{debug_prefix}_orig")

                    if seg["need_resize"]:
                        video_frames = [
                            self._resize_and_crop(frame, VIDEO_RESOLUTION_BUCKETS)
                            for frame in video_frames
                        ]

                    pixel_values = [self.image_transform(frame) for frame in video_frames]
                    pixel_values = torch.stack(pixel_values).transpose(0, 1)  # [C, T, H, W]

                    if self.is_debug:
                        self._save_debug_media(pixel_values, f"{debug_prefix}_transformed")

                    gen_pixel_values.append(pixel_values)
                    tokens = self._get_gen_video_num_attention_tokens(pixel_values)
                    num_gen_attention_tokens += tokens
                    num_attention_tokens += tokens
                    has_gen_output = True
                    gen_paths.append(seg["rel_path"])

        # ---- Dropout for classifier-free guidance training ----
        # Mutually exclusive: all_dropout drops everything, text_dropout drops only text
        if self.dropout and has_gen_output:
            rand_val = random.random()
            if rand_val < self.all_dropout_rate:
                # Drop ALL conditions: text + VLM visual + DiT source/ref
                user_content = [{"type": "text", "text": ""}]
                source_pixel_values = []
                ref_pixel_values = []
            elif rand_val < self.all_dropout_rate + self.text_dropout_rate:
                # Drop only text, keep visual conditions intact
                user_content = [
                    {"type": "text", "text": ""} if item["type"] == "text" else item
                    for item in user_content
                ]

        # ---- Align source and target dimensions ----
        if source_pixel_values and gen_pixel_values:
            for pair_idx, (src_pv, gen_pv) in enumerate(
                zip(source_pixel_values, gen_pixel_values)
            ):
                # Align source_video and target video lengths
                if src_pv.ndim == 4 and gen_pv.ndim == 4:
                    old_tokens = self._get_gen_video_num_attention_tokens(gen_pv)
                    new_src_pv = src_pv
                    new_gen_pv = gen_pv
                    src_num_frames = src_pv.shape[1]  # [C, T, H, W]
                    gen_num_frames = gen_pv.shape[1]
                    if src_num_frames != gen_num_frames:
                        min_frames = min(src_num_frames, gen_num_frames)
                        warnings.warn(
                            f"Source video (frames={src_num_frames}) and target video "
                            f"(frames={gen_num_frames}) have different lengths. "
                            f"Truncating both to {min_frames} frames."
                        )
                        new_src_pv = new_src_pv[:, :min_frames, :, :]
                        new_gen_pv = new_gen_pv[:, :min_frames, :, :]

                    src_height, src_width = new_src_pv.shape[-2:]
                    gen_height, gen_width = new_gen_pv.shape[-2:]
                    if src_height != gen_height or src_width != gen_width:
                        target_height, target_width = self._get_aligned_spatial_size(
                            src_height, src_width, gen_height, gen_width
                        )
                        warnings.warn(
                            f"Source video ({src_height}x{src_width}) and target video "
                            f"({gen_height}x{gen_width}) have different sizes. "
                            f"Resizing both to {target_height}x{target_width}."
                        )
                        if (src_height, src_width) != (target_height, target_width):
                            new_src_pv = self._resize_pixel_values_spatial(
                                new_src_pv, (target_height, target_width)
                            )
                        if (gen_height, gen_width) != (target_height, target_width):
                            new_gen_pv = self._resize_pixel_values_spatial(
                                new_gen_pv, (target_height, target_width)
                            )

                    if new_src_pv is not src_pv or new_gen_pv is not gen_pv:
                        source_pixel_values[pair_idx] = new_src_pv
                        gen_pixel_values[pair_idx] = new_gen_pv
                        new_tokens = self._get_gen_video_num_attention_tokens(new_gen_pv)
                        num_gen_attention_tokens += new_tokens - old_tokens
                        num_attention_tokens += new_tokens - old_tokens

                # Align source_image and target image sizes
                elif src_pv.ndim == 3 and gen_pv.ndim == 3:
                    old_tokens = self._get_gen_image_num_attention_tokens(gen_pv)
                    new_src_pv = src_pv
                    new_gen_pv = gen_pv
                    src_height, src_width = src_pv.shape[1], src_pv.shape[2]  # [C, H, W]
                    gen_height, gen_width = gen_pv.shape[1], gen_pv.shape[2]
                    if src_height != gen_height or src_width != gen_width:
                        target_height, target_width = self._get_aligned_spatial_size(
                            src_height, src_width, gen_height, gen_width
                        )
                        warnings.warn(
                            f"Source image ({src_height}x{src_width}) and target image "
                            f"({gen_height}x{gen_width}) have different sizes. "
                            f"Resizing both to {target_height}x{target_width}."
                        )
                        if (src_height, src_width) != (target_height, target_width):
                            new_src_pv = self._resize_pixel_values_spatial(
                                new_src_pv, (target_height, target_width)
                            )
                        if (gen_height, gen_width) != (target_height, target_width):
                            new_gen_pv = self._resize_pixel_values_spatial(
                                new_gen_pv, (target_height, target_width)
                            )

                    if new_src_pv is not src_pv or new_gen_pv is not gen_pv:
                        source_pixel_values[pair_idx] = new_src_pv
                        gen_pixel_values[pair_idx] = new_gen_pv
                        new_tokens = self._get_gen_image_num_attention_tokens(new_gen_pv)
                        num_gen_attention_tokens += new_tokens - old_tokens
                        num_attention_tokens += new_tokens - old_tokens

        # ---- Build tokenized inputs ----
        messages = [{"role": "user", "content": user_content}]
        if has_gen_output:
            add_generation_prompt = True
        else:
            messages.append({"role": "assistant", "content": assistant_content})
            add_generation_prompt = False

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_tensors="pt",
            return_dict=True,
        )
        input_ids = inputs["input_ids"].squeeze(0)
        num_attention_tokens += len(input_ids)

        # Build labels: mask everything except assistant response for understanding tasks
        labels = torch.full_like(input_ids, -100)
        if not has_gen_output:
            input_list = input_ids.tolist()
            start_idx = 0
            for i in reversed(range(len(input_list) - 1)):
                if input_list[i] == self.im_start_id and input_list[i + 1] == self.assistant_id:
                    start_idx = i + 2
                    break
            labels[start_idx:] = input_ids[start_idx:]

        return {
            "inputs": inputs,
            "labels": labels,
            "gen_pixel_values": gen_pixel_values,
            "source_pixel_values": source_pixel_values,
            "ref_pixel_values": ref_pixel_values,
            "prompts": prompts,
            "num_attention_tokens": num_attention_tokens,
            "num_gen_attention_tokens": num_gen_attention_tokens,
            "task": "gen" if has_gen_output else "und",
            # Debug for training
            "dataset_name": self.dataset_names[dataset_index],
            "data_info": data_info,
            # Debug for generation
            "und_paths": und_paths,
            "und_types": und_types,
            "gen_paths": gen_paths,
        }

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError
        self._internal_counter += 1
        if self._internal_counter % 200 == 0:
            gc.collect()

        dataset_index, data_index = self._get_data_index(idx)
        try:
            return self._getitem(dataset_index, data_index)
        except Exception as e:
            print(
                f"[Warning] Skipped bad sample {data_index} from "
                f"{self.dataset_names[dataset_index]}: {repr(e)}"
            )
            return None


class UniEvalDataset(UniTrainDataset):
    """
    Evaluation dataset that loads generated outputs alongside ground-truth data.
    """

    def __init__(
        self,
        dataset_config,
        config,
        gen_output_root: str,
    ):
        super().__init__(dataset_config=dataset_config, config=config, is_debug=False)
        self.gen_output_root = gen_output_root

    def _load_local_media(self, path):
        """Load image or video from a local path, returning normalized numpy array."""
        image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")
        if path.lower().endswith(image_extensions):
            image = np.array(Image.open(path).convert("RGB"))
            return np.expand_dims(image, axis=0) / 255.0
        vr = decord.VideoReader(path, ctx=decord.cpu(0))
        batch = vr.get_batch(list(range(len(vr))))
        video_array = batch.numpy() if isinstance(batch, torch.Tensor) else batch.asnumpy()
        return video_array / 255.0

    def _find_local_path(self, local_dir, rel_path):
        """Find the actual generated file, handling .mp4/.jpg extension variants."""
        stem = os.path.splitext(os.path.join(local_dir, rel_path))[0]
        for ext in (".mp4", ".jpg"):
            candidate = stem + ext
            if os.path.exists(candidate):
                return candidate
        return os.path.join(local_dir, rel_path)

    def _getitem(self, dataset_index, data_index):
        dataset_info = self.dataset_list[dataset_index]
        dataset_name = self.dataset_names[dataset_index]

        data_json_dir = dataset_info.data_json_dir
        json_path = os.path.join(data_json_dir, f"{data_index}.json")
        data_info = self._read_json(json_path)
        data_info["_data_index"] = data_index

        processor_fn = getattr(processors, dataset_info.process_func_name)
        segments = processor_fn(dataset_info, data_info)

        local_dir = os.path.join(self.gen_output_root, dataset_name, "outputs")

        instruction = ""
        ref_pixel_values = []
        source_pixel_values = []
        gen_pixel_values = []
        gen_paths = []
        und_paths = []
        und_types = []

        for seg in segments:
            if seg["is_target"] is False:
                if seg["type"] == "text":
                    instruction = seg["content"]
                    if "save_path" in seg:
                        gen_paths.append(seg["save_path"])

                elif seg["type"] == "image":
                    image = self._read_image(seg["path"])
                    if seg["need_resize"]:
                        image = self._resize_and_crop(image, IMAGE_RESOLUTION_BUCKETS)
                    ref_pixel_values.append(self.image_transform(image))
                    und_paths.append(seg["rel_path"])
                    und_types.append("ref")

                elif seg["type"] == "video":
                    raise NotImplementedError(
                        "Reference video (is_target=False, type=video) is not supported yet."
                    )

                elif seg["type"] == "source_image":
                    image = self._read_image(seg["path"])
                    if seg["need_resize"]:
                        image = self._resize_and_crop(image, IMAGE_RESOLUTION_BUCKETS)
                    source_pixel_values.append(self.image_transform(image))
                    und_paths.append(seg["rel_path"])
                    und_types.append("source")
                    if "save_path" in seg:
                        gen_paths.append(seg["save_path"])

                elif seg["type"] == "source_video":
                    video_frames = self._read_video(seg["path"])
                    if seg["need_resize"]:
                        video_frames = [
                            self._resize_and_crop(frame, VIDEO_RESOLUTION_BUCKETS)
                            for frame in video_frames
                        ]
                    pixel_values = [self.image_transform(frame) for frame in video_frames]
                    pixel_values = torch.stack(pixel_values).transpose(0, 1)  # [C, T, H, W]
                    source_pixel_values.append(pixel_values)
                    und_paths.append(seg["rel_path"])
                    und_types.append("source")
                    if "save_path" in seg:
                        gen_paths.append(seg["save_path"])

            elif seg["is_target"] is True and seg["type"] in ("image", "video"):
                local_path = self._find_local_path(local_dir, seg["rel_path"])
                gen_pixel_values.append(self._load_local_media(local_path))
                gen_paths.append(local_path)

        return {
            "dataset_name": dataset_name,
            "instruction": instruction,
            "data_info": data_info,
            "ref_pixel_values": ref_pixel_values,
            "source_pixel_values": source_pixel_values,
            "und_paths": und_paths,
            "und_types": und_types,
            "gen_pixel_values": gen_pixel_values,
            "gen_paths": gen_paths,
        }

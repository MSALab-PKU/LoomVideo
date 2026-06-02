from typing import List, Dict, Any, Optional, Union
import math
import tqdm

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.models.embeddings import FP32SiLU
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers import UniPCMultistepScheduler
from diffusers.video_processor import VideoProcessor

from transformers import AutoProcessor, AutoTokenizer
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRMSNorm

from flash_attn import flash_attn_varlen_func

from src.models.utils import unfreeze_model
from .qwen3vl import Qwen3VLForConditionalGeneration
from .wan22 import WanTransformer3DModel


class PixArtAlphaTextProjectionNorm(nn.Module):
    """
    Projects caption embeddings. Also handles dropout for classifier-free guidance.

    Adapted from https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/nets/PixArt_blocks.py
    """

    def __init__(self, in_features: int, hidden_size: int, out_features: Optional[int] = None, act_fn: str = "gelu_tanh"):
        super().__init__()
        if out_features is None:
            out_features = hidden_size
        self.norm = Qwen3VLTextRMSNorm(in_features)
        self.linear_1 = nn.Linear(in_features=in_features, out_features=4 * hidden_size, bias=True)
        if act_fn == "gelu_tanh":
            self.act_1 = nn.GELU(approximate="tanh")
        elif act_fn == "silu":
            self.act_1 = nn.SiLU()
        elif act_fn == "silu_fp32":
            self.act_1 = FP32SiLU()
        else:
            raise ValueError(f"Unknown activation function: {act_fn}")
        self.linear_2 = nn.Linear(in_features=4 * hidden_size, out_features=out_features, bias=True)

    def forward(self, caption: torch.Tensor) -> torch.Tensor:
        hidden_states = self.norm(caption)
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.act_1(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class LoomVideo(ModelMixin):
    """
    LoomVideo: Unified multimodal model for controllable video generation.

    Architecture overview:
        - Understanding backbone: Qwen3-VL
        - Generation backbone: Wan2.2 DiT
        - Fusion mechanism: Layer-wise cross-attention from DiT to VLM hidden states
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Understanding model (Vision-Language Model)
        self.und_model = Qwen3VLForConditionalGeneration.from_pretrained(
            config.model.und.pretrained_model_path,
            dtype=torch.bfloat16,
        )
        self.und_model.requires_grad_(False)

        # Generation model (Diffusion Transformer)
        self.gen_vae = AutoencoderKLWan.from_pretrained(
            config.model.gen.pretrained_model_path,
            subfolder="vae",
            torch_dtype=torch.bfloat16,
        )
        self.gen_vae.requires_grad_(False)

        self.gen_model = WanTransformer3DModel.from_pretrained(
            config.model.gen.pretrained_model_path,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
        )
        self.gen_model.requires_grad_(False)

        unfreeze_model(self, self.config.model.trainable_modules)

        # Noise scheduler
        self.gen_scheduler = UniPCMultistepScheduler.from_pretrained(
            config.model.gen.pretrained_model_path,
            subfolder="scheduler",
        )

        # Cross-attention projection: VLM hidden states -> DiT conditioning
        und_attn_heads = self.und_model.model.language_model.config.num_attention_heads
        und_attn_head_dim = self.und_model.model.language_model.config.head_dim
        und_dim = und_attn_heads * und_attn_head_dim

        self.gen_model.mllm_embedder = PixArtAlphaTextProjectionNorm(und_dim, self.gen_model.inner_dim, act_fn="gelu_tanh")
        nn.init.zeros_(self.gen_model.mllm_embedder.linear_2.weight)
        if self.gen_model.mllm_embedder.linear_2.bias is not None:
            nn.init.zeros_(self.gen_model.mllm_embedder.linear_2.bias)

        # Source video conditioning embedding (zero-initialized for stable training)
        if config.model.gen.use_source_embedding:
            patch_size = self.gen_model.config.patch_size
            in_channels = self.gen_model.config.in_channels
            inner_dim = self.gen_model.inner_dim

            self.gen_model.source_patch_embedding = nn.Conv3d(
                in_channels, inner_dim, kernel_size=patch_size, stride=patch_size
            )
            nn.init.zeros_(self.gen_model.source_patch_embedding.weight)
            if self.gen_model.source_patch_embedding.bias is not None:
                nn.init.zeros_(self.gen_model.source_patch_embedding.bias)

        # Pipeline for T5 text encoding
        self.pipe = WanPipeline.from_pretrained(
            self.config.model.gen.pretrained_model_path,
            transformer=self.gen_model,
            vae=self.gen_vae,
            torch_dtype=torch.bfloat16,
        )
        self.text_encoder = self.pipe.text_encoder
        self.text_encoder.requires_grad_(False)
        self.text_encoder_max_sequence_length = 512

        # VAE scale factors
        self.vae_scale_factor_temporal = self.pipe.vae_scale_factor_temporal
        self.vae_scale_factor_spatial = self.pipe.vae_scale_factor_spatial
        self.gen_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

        # Training settings
        self.gradient_checkpointing = config.model.gradient_checkpointing
        self.base_timestep_shift = config.model.gen.base_timestep_shift
        self.num_attn_token_base_shift = config.model.gen.num_attn_token_base_shift
        self.timestep_shift_scale = config.model.gen.timestep_shift_scale

        # Pre-compute fixed T5 embeddings for training
        fixed_prompt = "a photo of"
        device = self.device
        dtype = self.gen_model.dtype

        with torch.no_grad():
            self.fixed_t5_embeds = self.pipe._get_t5_prompt_embeds(
                prompt=[fixed_prompt],
                max_sequence_length=10,
                device=device,
                dtype=dtype,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.und.pretrained_model_path, trust_remote_code=True
        )

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Return all parameters that require gradient updates."""
        return [p for p in self.parameters() if p.requires_grad]

    def get_trainable_components(self) -> Dict[str, nn.Module]:
        """Return a dict of top-level components that contain trainable parameters."""
        components = {}
        if any(p.requires_grad for p in self.gen_model.parameters()):
            components["gen_model"] = self.gen_model
        if any(p.requires_grad for p in self.und_model.parameters()):
            components["und_model"] = self.und_model
        return components

    def get_latents(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Encode pixel values into normalized VAE latents.

        Args:
            pixel_values: Input tensor of shape [C, H, W] (image) or [C, T, H, W] (video).

        Returns:
            Normalized latent tensor of shape [1, Z, T', H', W'].
        """
        if pixel_values.dim() == 3:
            pixel_values = pixel_values.unsqueeze(1).unsqueeze(0)
        elif pixel_values.dim() == 4:
            pixel_values = pixel_values.unsqueeze(0)
        else:
            raise ValueError("pixel_values must be a 3D or 4D tensor")

        pixel_values = pixel_values.to(dtype=self.gen_vae.dtype)

        self.gen_vae.eval()
        with torch.no_grad():
            posterior = self.gen_vae.encode(pixel_values).latent_dist
            latents = posterior.sample()

            # Normalize latents using VAE statistics
            latents_mean = (
                torch.tensor(self.gen_vae.config.latents_mean)
                .view(1, self.gen_vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.gen_vae.config.latents_std).view(
                1, self.gen_vae.config.z_dim, 1, 1, 1
            ).to(latents.device, latents.dtype)

            latents = (latents - latents_mean) * latents_std

        latents = latents.to(dtype=self.gen_model.dtype)
        return latents

    def compute_gen_loss(self, result_dict: Dict[str, torch.Tensor], mode: str = "sum") -> torch.Tensor:
        """
        Compute MSE loss between predicted and target flow vectors.

        Args:
            result_dict: Dict with 'pred' and 'target' tensors.
            mode: 'sum' for element-wise sum, 'mean' for per-sample mean.

        Returns:
            Scalar loss tensor.
        """
        pred, target = result_dict["pred"], result_dict["target"]
        if mode == "sum":
            loss = ((pred.float() - target.float()) ** 2).sum()
        elif mode == "mean":
            loss = ((pred.float() - target.float()) ** 2).reshape(target.shape[0], -1).mean(dim=1).mean()
        else:
            raise ValueError(f"Unknown loss mode: {mode}. Expected 'sum' or 'mean'.")
        return loss

    def get_dynamic_shift(self, num_attention_tokens: float) -> float:
        """
        Compute resolution-aware timestep shift for flow matching.

        Args:
            num_attention_tokens: Number of latent tokens in the generation sample.

        Returns:
            Dynamic shift value clamped to [base_shift, 8.0].
        """
        ratio = num_attention_tokens / self.num_attn_token_base_shift
        if ratio <= 0:
            return self.base_timestep_shift
        dynamic_shift = self.base_timestep_shift + math.log2(ratio) * self.timestep_shift_scale
        return max(self.base_timestep_shift, min(dynamic_shift, 8.0))

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int = 16,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Prepare initial Gaussian noise latents for the diffusion process.

        Args:
            batch_size: Number of samples to generate.
            num_channels_latents: Latent channel dimension.
            height: Target video height in pixels.
            width: Target video width in pixels.
            num_frames: Target number of video frames.
            dtype: Desired tensor dtype.
            device: Target device.
            generator: Optional random generator for reproducibility.
            latents: Pre-computed latents (returned as-is if provided).

        Returns:
            Noise latent tensor of shape [B, C, T', H', W'].
        """
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        shape = (
            batch_size,
            num_channels_latents,
            num_latent_frames,
            int(height) // self.vae_scale_factor_spatial,
            int(width) // self.vae_scale_factor_spatial,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"Generator list length {len(generator)} does not match batch size {batch_size}."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return latents

    def prepare_qwen_negative_prompt(self, config, negative_prompt: Optional[str]) -> Dict[str, torch.Tensor]:
        """
        Build unconditional MLLM inputs for classifier-free guidance.

        Args:
            config: Model config with pretrained model path.
            negative_prompt: Optional negative text prompt.

        Returns:
            Dict of tokenized inputs ready for the understanding model.
        """
        processor = AutoProcessor.from_pretrained(config.model.und.pretrained_model_path)

        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": f"generate an image: {negative_prompt}"}],
            }
        ]

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.und_model.device)

        return inputs

    def prepare_qwen_visual_only_prompt(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Build MLLM input with visual tokens only (empty text) for visual-only CFG branch.

        Extracts vision token spans (<|vision_start|>...<|vision_end|>) from the original
        input_ids and wraps them in a minimal chat template, preserving pixel values and
        grid metadata for the vision encoder.

        Args:
            input_ids: Original tokenized input containing vision placeholders.
            pixel_values: Optional image pixel values.
            pixel_values_videos: Optional video pixel values.
            image_grid_thw: Optional image grid dimensions.
            video_grid_thw: Optional video grid dimensions.

        Returns:
            Dict of model inputs with visual tokens but no text content.
        """
        vision_start_id = self.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        vision_end_id = self.tokenizer.convert_tokens_to_ids("<|vision_end|>")
        im_start_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        nl_id = self.tokenizer.encode("\n", add_special_tokens=False)[-1]
        user_id = self.tokenizer.convert_tokens_to_ids("user")
        assistant_id = self.tokenizer.convert_tokens_to_ids("assistant")

        # Extract all vision spans from original input_ids
        ids = input_ids.squeeze().tolist()
        vision_spans = []
        i = 0
        while i < len(ids):
            if ids[i] == vision_start_id:
                j = i + 1
                while j < len(ids) and ids[j] != vision_end_id:
                    j += 1
                vision_spans.extend(ids[i:j + 1])
                i = j + 1
            else:
                i += 1

        # Construct: <|im_start|>user\n{vision_tokens}<|im_end|>\n<|im_start|>assistant\n
        new_ids = [im_start_id, user_id, nl_id] + vision_spans + [im_end_id, nl_id, im_start_id, assistant_id, nl_id]
        new_input_ids = torch.tensor([new_ids], dtype=input_ids.dtype, device=input_ids.device)
        new_attention_mask = torch.ones_like(new_input_ids)

        result = {
            "input_ids": new_input_ids,
            "attention_mask": new_attention_mask,
        }
        if pixel_values is not None:
            result["pixel_values"] = pixel_values
        if pixel_values_videos is not None:
            result["pixel_values_videos"] = pixel_values_videos
        if image_grid_thw is not None:
            result["image_grid_thw"] = image_grid_thw
        if video_grid_thw is not None:
            result["video_grid_thw"] = video_grid_thw

        return result

    def get_flash_attn_kwargs(
        self,
        und_cu_seq_lens: List[int],
        gen_cu_seq_lens: List[int],
        gen_sample_index: List[int],
    ):
        """
        Compute Flash Attention metadata for cross-attention between DiT and VLM.

        Maps each generation sample to its corresponding understanding sequence,
        computing cumulative sequence lengths and max lengths for variable-length
        flash attention.

        Args:
            und_cu_seq_lens: Cumulative sequence lengths for understanding tokens.
            gen_cu_seq_lens: Cumulative sequence lengths for generation tokens.
            gen_sample_index: Mapping from generation samples to understanding samples.

        Returns:
            Tuple of (flash_attn_kwargs dict, und_seq_indices tensor).
        """
        und_seq_indices = []
        gen_max_seqlen = 0
        und_max_seqlen = 0
        matched_und_cu_seq_lens = [0]

        for i, idx in enumerate(gen_sample_index):
            u_start = und_cu_seq_lens[idx]
            u_end = und_cu_seq_lens[idx + 1]

            und_seq_indices.append(torch.arange(u_start, u_end, device=self.device))

            u_len = u_end - u_start
            matched_und_cu_seq_lens.append(matched_und_cu_seq_lens[-1] + u_len)
            und_max_seqlen = max(und_max_seqlen, u_len)

            g_len = gen_cu_seq_lens[i + 1] - gen_cu_seq_lens[i]
            if isinstance(g_len, torch.Tensor):
                g_len = g_len.item()
            gen_max_seqlen = max(gen_max_seqlen, g_len)

        if len(und_seq_indices) > 0:
            und_seq_indices = torch.cat(und_seq_indices)
        else:
            und_seq_indices = torch.empty(0, dtype=torch.long, device=self.device)

        cu_seqlens_q = (
            torch.tensor(gen_cu_seq_lens, dtype=torch.int32, device=self.device)
            if not isinstance(gen_cu_seq_lens, torch.Tensor)
            else gen_cu_seq_lens.to(dtype=torch.int32, device=self.device)
        )
        cu_seqlens_k = torch.tensor(matched_und_cu_seq_lens, dtype=torch.int32, device=self.device)

        flash_attn_kwargs = {
            "cu_seqlens_q": cu_seqlens_q,
            "cu_seqlens_k": cu_seqlens_k,
            "max_seqlen_q": gen_max_seqlen,
            "max_seqlen_k": und_max_seqlen,
        }

        return flash_attn_kwargs, und_seq_indices

    def _joint_block_forward(
        self,
        und_hidden_states: torch.Tensor,
        gen_hidden_states: torch.Tensor,
        gen_block: nn.Module,
        gen_timestep_proj: torch.Tensor,
        gen_rotary_emb: torch.Tensor,
        flash_attn_kwargs: Dict[str, Any],
        und_seq_indices: torch.Tensor,
        gen_cu_seq_lens: List[int],
    ) -> torch.Tensor:
        """
        Execute one joint transformer block with DiT self-attention + cross-attention to VLM.

        Each block performs:
            1. DiT self-attention on generation tokens
            2. Cross-attention from generation tokens (Q) to VLM hidden states (KV)
            3. DiT feed-forward network

        Args:
            und_hidden_states: VLM hidden states for the current layer.
            gen_hidden_states: DiT hidden states.
            gen_block: The DiT transformer block.
            gen_timestep_proj: Timestep projection for adaptive normalization.
            gen_rotary_emb: Rotary position embeddings.
            flash_attn_kwargs: Flash attention configuration.
            und_seq_indices: Indices to select matched VLM tokens.
            gen_cu_seq_lens: Cumulative sequence lengths for generation.

        Returns:
            Updated generation hidden states.
        """
        # Select VLM tokens corresponding to each generation sample
        matched_und_hidden_states = und_hidden_states[:, und_seq_indices, :]
        embedded_und_hidden_states = self.gen_model.mllm_embedder(matched_und_hidden_states)

        # DiT self-attention
        gen_hidden_states = gen_block.forward_selfattn(
            gen_hidden_states, gen_timestep_proj, gen_rotary_emb, cu_seq_lens=gen_cu_seq_lens
        )

        # Cross-attention: generation queries attend to VLM keys/values
        gen_norm_hidden_states = gen_block.norm2(gen_hidden_states.float()).type_as(gen_hidden_states)
        gen_q, gen_k, gen_v, _, _ = gen_block.attn2.processor.get_qkv(
            gen_block.attn2, gen_norm_hidden_states, embedded_und_hidden_states, None
        )

        # Reshape for flash attention: [1, num_heads, L, head_dim] -> [L, num_heads, head_dim]
        gen_q = gen_q.squeeze(0).transpose(0, 1).contiguous()
        gen_k = gen_k.squeeze(0).transpose(0, 1).contiguous()
        gen_v = gen_v.squeeze(0).transpose(0, 1).contiguous()

        if gen_q.shape[0] == 0:
            gen_attn_output = torch.zeros_like(gen_q)
        else:
            gen_attn_output = flash_attn_varlen_func(
                gen_q,
                gen_k,
                gen_v,
                cu_seqlens_q=flash_attn_kwargs["cu_seqlens_q"],
                cu_seqlens_k=flash_attn_kwargs["cu_seqlens_k"],
                max_seqlen_q=flash_attn_kwargs["max_seqlen_q"],
                max_seqlen_k=flash_attn_kwargs["max_seqlen_k"],
                dropout_p=0.0,
                causal=False,
            )

        gen_attn_output = gen_attn_output.flatten(1, 2).unsqueeze(0)
        gen_attn_output = gen_block.attn2.to_out[0](gen_attn_output)
        gen_attn_output = gen_block.attn2.to_out[1](gen_attn_output)

        # Feed-forward network
        gen_hidden_states = gen_block.forward_crossattn_later_layer(
            gen_hidden_states,
            gen_attn_output,
            gen_timestep_proj,
        )

        return gen_hidden_states

    def forward_loss(self, batch: Dict[str, Any]):
        """
        Compute training loss using flow matching objective.

        Args:
            batch: Training batch containing inputs, gen_pixel_values,
                   source_pixel_values, ref_pixel_values, etc.

        Returns:
            Tuple of (und_loss, gen_loss) tensors.
        """
        targets = []
        noised_hidden_states = []
        source_hidden_states = []
        ref_hidden_states = []

        valid_gen_indices = [idx for idx, gen_pv in enumerate(batch["gen_pixel_values"]) if len(gen_pv) > 0]
        num_gen_samples = len(valid_gen_indices)

        if num_gen_samples > 0:
            if "t_step" in batch:
                t = batch["t_step"].to(device=self.gen_model.device)
                if t.dim() == 0:
                    t = t.unsqueeze(0).expand(num_gen_samples)
                elif t.shape[0] == len(batch["gen_pixel_values"]):
                    t = t[valid_gen_indices]
            else:
                # Sample timesteps with resolution-aware dynamic shifting
                t_logit = torch.exp(torch.randn(num_gen_samples, device=self.gen_model.device))
                t = t_logit / (t_logit + 1)

                raw_tokens_tensor = torch.tensor(
                    batch["num_gen_attention_tokens"], device=self.gen_model.device, dtype=torch.float32
                )
                tokens_for_gen = raw_tokens_tensor[valid_gen_indices].to(dtype=torch.float32)
                shifts = []
                for num_tokens in tokens_for_gen.tolist():
                    shifts.append(self.get_dynamic_shift(float(num_tokens)))
                shifts = torch.tensor(shifts, device=self.gen_model.device, dtype=torch.float32)
                t = (shifts * t) / (1.0 + (shifts - 1.0) * t)
        else:
            t = torch.empty(0, device=self.gen_model.device)

        t_expand_batch = t[:, None, None, None, None]

        gen_idx = 0
        for idx, gen_pixel_values in enumerate(batch["gen_pixel_values"]):
            if len(gen_pixel_values) == 0:
                # Understanding-only sample (no generation target)
                targets.append(None)
                noised_hidden_states.append(None)
                source_hidden_states.append(None)
                ref_hidden_states.append(None)
            else:
                assert len(gen_pixel_values) == 1, "Only one generation target per sample is supported."
                assert len(batch["source_pixel_values"][idx]) <= 1, "At most one source video per sample is supported."

                latents = self.get_latents(gen_pixel_values[0])
                z_1 = torch.randn_like(latents)
                eps = 1e-3

                current_t_expand = t_expand_batch[gen_idx]

                # Flow matching interpolation: z_t = (1-t)*data + (eps + (1-eps)*t)*noise
                z_t = (1 - current_t_expand) * latents + (eps + (1 - eps) * current_t_expand) * z_1
                # Target velocity: u = (1-eps)*noise - data
                target_velocity = (1 - eps) * z_1 - latents

                z_t = z_t.to(latents.dtype)
                target_velocity = target_velocity.to(latents.dtype)

                targets.append(target_velocity)
                noised_hidden_states.append(z_t)

                # Source conditioning
                source_pixel_values = batch["source_pixel_values"][idx]
                if len(source_pixel_values) > 0:
                    source_hidden_states.append(self.get_latents(source_pixel_values[0]))
                else:
                    source_hidden_states.append(torch.zeros_like(latents))

                # Reference conditioning
                ref_pvs = batch["ref_pixel_values"][idx]
                if len(ref_pvs) > 0:
                    ref_hidden_states.append([self.get_latents(rpv) for rpv in ref_pvs])
                else:
                    ref_hidden_states.append(None)

                gen_idx += 1

        gen_timestep = (t * self.gen_scheduler.config.num_train_timesteps).to(self.gen_model.dtype)
        gen_encoder_hidden_states = self.fixed_t5_embeds.to(self.gen_model.device).expand(num_gen_samples, -1, -1)

        und_hidden_states, pred = self.forward(
            inputs=batch["inputs"],
            gen_hidden_states=noised_hidden_states,
            gen_timestep=gen_timestep,
            gen_encoder_hidden_states=gen_encoder_hidden_states,
            source_hidden_states=source_hidden_states,
            source_scale=t,
            ref_hidden_states=ref_hidden_states,
        )

        # Understanding loss (placeholder for future use)
        und_loss = torch.zeros((), device=self.device, dtype=torch.float32, requires_grad=True)

        # Generation loss (MSE on flow vectors)
        gen_loss = torch.zeros((), device=self.device, dtype=torch.float32, requires_grad=True)
        total_gen_elements = 0

        gen_count = 0
        for target in targets:
            if target is not None:
                current_sample_loss = self.compute_gen_loss({"pred": pred[gen_count], "target": target})
                gen_loss = gen_loss + current_sample_loss
                total_gen_elements += target.numel()
                gen_count += 1

        if total_gen_elements > 0:
            gen_loss = gen_loss / total_gen_elements

        return und_loss, gen_loss

    def forward(
        self,
        inputs: List[Dict[str, Any]],
        gen_hidden_states: List[torch.Tensor],
        gen_timestep: torch.LongTensor,
        gen_encoder_hidden_states: torch.Tensor,
        source_hidden_states: Optional[List[torch.Tensor]] = None,
        source_scale: Optional[torch.Tensor] = None,
        ref_hidden_states: Optional[List[torch.Tensor]] = None,
    ):
        """
        Full forward pass with layer-wise cross-attention fusion.

        Args:
            inputs: List of tokenized VLM inputs (one per batch sample).
            gen_hidden_states: List of noised latent tensors (None for non-generation samples).
            gen_timestep: Diffusion timestep for each generation sample.
            gen_encoder_hidden_states: T5 text encoder hidden states.
            source_hidden_states: Optional source video latents for conditioning.
            source_scale: Timestep-dependent source conditioning scale.
            ref_hidden_states: Optional reference image/video latents.

        Returns:
            Tuple of (und_hidden_states, gen_hidden_states) after processing.
        """
        # Compute cumulative sequence lengths for understanding tokens
        und_cu_seq_lens = [0]
        for inp in inputs:
            length = inp["input_ids"].shape[-1]
            und_cu_seq_lens.append(und_cu_seq_lens[-1] + length)

        # VLM has more layers than DiT; early VLM layers run independently
        num_early_layers = len(self.und_model.model.language_model.layers) - len(self.gen_model.blocks)

        # Run VLM forward to get all layer hidden states
        batch_all_und_hidden_states = []
        for inp in inputs:
            und_outputs = self.und_model.model.forward(**inp)
            all_und_hidden_states = und_outputs["hidden_states"]
            batch_all_und_hidden_states.append(all_und_hidden_states)

        # Transpose: [batch, layers] -> [layers, batch] and concatenate along sequence dim
        transposed_layers = zip(*batch_all_und_hidden_states)
        layer_wise_hidden_states = [torch.cat(layer_samples, dim=1) for layer_samples in transposed_layers]

        # DiT early layers (patch embedding, timestep embedding, etc.)
        (
            gen_hidden_states,
            gen_encoder_hidden_states,
            gen_timestep_proj,
            gen_rotary_emb,
            gen_temb,
            gen_shape_list,
            gen_cu_seq_lens,
            gen_sample_index,
            gen_seq_lens,
        ) = self.gen_model.forward_early_layers(
            hidden_states=gen_hidden_states,
            timestep=gen_timestep,
            encoder_hidden_states=gen_encoder_hidden_states,
            source_hidden_states=source_hidden_states,
            source_scale=source_scale,
            ref_hidden_states=ref_hidden_states,
        )

        # Compute flash attention metadata for cross-attention
        flash_attn_kwargs, und_seq_indices = self.get_flash_attn_kwargs(
            und_cu_seq_lens,
            gen_cu_seq_lens,
            gen_sample_index,
        )

        # Layer-wise cross-attention blocks
        for index, gen_block in enumerate(self.gen_model.blocks):
            und_layer_idx = -1 if self.config.model.und.only_last_hidden_states else index + num_early_layers

            if self.training and self.gradient_checkpointing:
                gen_hidden_states = checkpoint(
                    self._joint_block_forward,
                    layer_wise_hidden_states[und_layer_idx],
                    gen_hidden_states,
                    gen_block,
                    gen_timestep_proj,
                    gen_rotary_emb,
                    flash_attn_kwargs,
                    und_seq_indices,
                    gen_cu_seq_lens,
                    use_reentrant=False,
                )
            else:
                gen_hidden_states = self._joint_block_forward(
                    layer_wise_hidden_states[und_layer_idx],
                    gen_hidden_states,
                    gen_block,
                    gen_timestep_proj,
                    gen_rotary_emb,
                    flash_attn_kwargs,
                    und_seq_indices,
                    gen_cu_seq_lens,
                )

        # Final output projections
        und_hidden_states = self.und_model.model.language_model.get_output(layer_wise_hidden_states[-1])
        gen_hidden_states = self.gen_model.get_output(
            gen_hidden_states, gen_temb, gen_cu_seq_lens, gen_shape_list, gen_seq_lens
        )

        return und_hidden_states, gen_hidden_states

    def forward_gen(
        self,
        all_und_hidden_states: List[torch.Tensor],
        gen_hidden_states: torch.Tensor,
        gen_timestep: torch.LongTensor,
        gen_encoder_hidden_states: torch.Tensor,
        source_hidden_states: Optional[torch.Tensor] = None,
        source_scale: Optional[torch.Tensor] = None,
        ref_hidden_states: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Generation-only forward pass used during inference.

        Unlike the training forward pass, this takes pre-computed VLM hidden states
        and runs only the DiT with cross-attention, avoiding redundant VLM computation
        across denoising steps.

        Args:
            all_und_hidden_states: Pre-computed per-layer VLM hidden states.
            gen_hidden_states: Current noisy latent tensor.
            gen_timestep: Current diffusion timestep.
            gen_encoder_hidden_states: T5 text encoder hidden states.
            source_hidden_states: Optional source video latents.
            source_scale: Timestep-dependent source scale.
            ref_hidden_states: Optional reference latents.

        Returns:
            Predicted noise/velocity tensor.
        """
        num_early_layers = len(self.und_model.model.language_model.layers) - len(self.gen_model.blocks)

        # DiT early layers
        (
            gen_hidden_states,
            gen_encoder_hidden_states,
            gen_timestep_proj,
            gen_rotary_emb,
            gen_temb,
            gen_shape_list,
            gen_cu_seq_lens,
            _,
            gen_seq_lens,
        ) = self.gen_model.forward_early_layers(
            hidden_states=[gen_hidden_states],
            timestep=gen_timestep,
            encoder_hidden_states=gen_encoder_hidden_states,
            source_hidden_states=[source_hidden_states],
            source_scale=source_scale,
            ref_hidden_states=[ref_hidden_states],
        )

        # Cross-attention: DiT attends to VLM hidden states at each layer
        for index, gen_block in enumerate(self.gen_model.blocks):
            und_layer_idx = -1 if self.config.model.und.only_last_hidden_states else num_early_layers + index
            und_hidden_states = all_und_hidden_states[und_layer_idx]
            embedded_und_hidden_states = self.gen_model.mllm_embedder(und_hidden_states)

            gen_hidden_states = gen_block(
                gen_hidden_states,
                embedded_und_hidden_states,
                gen_timestep_proj,
                gen_rotary_emb,
            )

        gen_hidden_states = self.gen_model.get_output(
            gen_hidden_states, gen_temb, gen_cu_seq_lens, gen_shape_list, gen_seq_lens
        )

        return gen_hidden_states[0]

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        negative_prompt: Optional[str] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 121,
        num_inference_steps: int = 50,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        guidance_scale: float = 5.0,
        guidance_scale_visual: float = 2.0,
        source_pixel_values: Optional[List[torch.Tensor]] = None,
        ref_pixel_values: Optional[List[torch.Tensor]] = None,
    ):
        """
        Generate video using cascaded classifier-free guidance.

        Supports three-level CFG:
            1. Unconditional (no text, no visual, no source, no ref)
            2. Visual-only (visual tokens only, no text description)
            3. Full condition (text + visual + source + ref)

        Args:
            input_ids: Tokenized text input.
            attention_mask: Attention mask for text input.
            pixel_values: Image pixel values for visual conditioning.
            pixel_values_videos: Video pixel values for visual conditioning.
            image_grid_thw: Image grid dimensions (T, H, W).
            video_grid_thw: Video grid dimensions (T, H, W).
            negative_prompt: Negative prompt for unconditional branch.
            height: Output video height in pixels.
            width: Output video width in pixels.
            num_frames: Number of output video frames.
            num_inference_steps: Number of denoising steps.
            generator: Random generator for reproducibility.
            guidance_scale: Text guidance scale (s_t).
            guidance_scale_visual: Visual guidance scale (s_v).
            source_pixel_values: Source video for temporal conditioning.
            ref_pixel_values: Reference images/videos for appearance conditioning.

        Returns:
            Generated video as numpy array of shape [T, H, W, C].
        """
        device = self.gen_model.device
        do_classifier_free_guidance = (guidance_scale > 1.0 or guidance_scale_visual > 1.0)

        # Initialize Gaussian noise latents
        gen_latents = self.prepare_latents(
            batch_size=1,
            num_channels_latents=self.gen_model.config.in_channels,
            height=height,
            width=width,
            num_frames=num_frames,
            dtype=self.dtype,
            device=device,
            generator=generator,
        )

        # Encode source video to latent space
        if source_pixel_values is not None:
            source_latents = self.get_latents(source_pixel_values[0])
        else:
            source_latents = torch.zeros_like(gen_latents)

        # Encode reference images/videos to latent space
        if ref_pixel_values is not None and len(ref_pixel_values) > 0:
            ref_latents = [self.get_latents(rpv) for rpv in ref_pixel_values]
        else:
            ref_latents = None

        # Prepare timesteps
        self.gen_scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.gen_scheduler.timesteps

        # Full condition: text + visual
        outputs_full = self.und_model.model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
        )
        all_hidden_states_full = outputs_full["hidden_states"]

        if do_classifier_free_guidance:
            # Visual-only condition: visual tokens, no text
            has_visual_input = (pixel_values is not None or pixel_values_videos is not None)
            if has_visual_input:
                visual_only_inputs = self.prepare_qwen_visual_only_prompt(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    pixel_values_videos=pixel_values_videos,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                )
                outputs_visual = self.und_model.model.forward(**visual_only_inputs)
                all_hidden_states_visual = outputs_visual["hidden_states"]
            else:
                all_hidden_states_visual = None

            # Unconditional: empty text, no visual, no source, no ref
            uncond_inputs = self.prepare_qwen_negative_prompt(self.config, negative_prompt)
            outputs_uncond = self.und_model.model.forward(**uncond_inputs)
            all_hidden_states_uncond = outputs_uncond["hidden_states"]

        # Denoising loop
        progress_bar = tqdm.tqdm(range(num_inference_steps), disable=False)

        for t in timesteps:
            gen_latents_input = self.gen_scheduler.scale_model_input(gen_latents, t)
            gen_timestep = t.expand(gen_latents_input.shape[0])

            # Compute source scale from current timestep
            current_sigma = t.float() / self.gen_scheduler.config.num_train_timesteps
            current_source_scale = current_sigma.unsqueeze(0).to(device)

            # Full condition forward
            gen_noise_full = self.forward_gen(
                all_und_hidden_states=all_hidden_states_full,
                gen_hidden_states=gen_latents_input,
                gen_timestep=gen_timestep,
                gen_encoder_hidden_states=self.fixed_t5_embeds.to(device),
                source_hidden_states=source_latents,
                source_scale=current_source_scale,
                ref_hidden_states=ref_latents,
            )

            if do_classifier_free_guidance:
                # Unconditional forward
                gen_noise_uncond = self.forward_gen(
                    all_und_hidden_states=all_hidden_states_uncond,
                    gen_hidden_states=gen_latents_input,
                    gen_timestep=gen_timestep,
                    gen_encoder_hidden_states=self.fixed_t5_embeds.to(device),
                    source_hidden_states=torch.zeros_like(gen_latents),
                    source_scale=current_source_scale,
                    ref_hidden_states=None,
                )

                if all_hidden_states_visual is not None:
                    # Visual-only condition forward
                    gen_noise_visual = self.forward_gen(
                        all_und_hidden_states=all_hidden_states_visual,
                        gen_hidden_states=gen_latents_input,
                        gen_timestep=gen_timestep,
                        gen_encoder_hidden_states=self.fixed_t5_embeds.to(device),
                        source_hidden_states=source_latents,
                        source_scale=current_source_scale,
                        ref_hidden_states=ref_latents,
                    )

                    # Cascaded CFG
                    gen_noise_pred = (
                        gen_noise_uncond
                        + guidance_scale_visual * (gen_noise_visual - gen_noise_uncond)
                        + guidance_scale * (gen_noise_full - gen_noise_visual)
                    )
                else:
                    # No visual input: standard CFG
                    gen_noise_pred = gen_noise_uncond + guidance_scale * (gen_noise_full - gen_noise_uncond)
            else:
                gen_noise_pred = gen_noise_full

            gen_latents = self.gen_scheduler.step(gen_noise_pred, t, gen_latents, return_dict=False)[0]
            progress_bar.update(1)

        # Decode latents to pixel space
        gen_latents = gen_latents.to(self.gen_vae.dtype)
        gen_latents_mean = (
            torch.tensor(self.gen_vae.config.latents_mean)
            .view(1, self.gen_vae.config.z_dim, 1, 1, 1)
            .to(gen_latents.device, gen_latents.dtype)
        )
        gen_latents_std = (
            1.0 / torch.tensor(self.gen_vae.config.latents_std)
            .view(1, self.gen_vae.config.z_dim, 1, 1, 1)
            .to(gen_latents.device, gen_latents.dtype)
        )
        gen_latents = gen_latents / gen_latents_std + gen_latents_mean
        output = self.gen_vae.decode(gen_latents, return_dict=False)[0]
        output = self.gen_processor.postprocess_video(output, output_type="np")

        return output[0]

    def get_val_loss_batch(
        self,
        batch: Dict[str, Any],
        num_valloss_timesteps: int = 20,
    ) -> List[Dict[str, float]]:
        """
        Compute validation loss across multiple noise levels for a batch.

        Args:
            batch: Collated validation batch.
            num_valloss_timesteps: Number of timesteps to evaluate.

        Returns:
            List of dicts (one per sample) with keys: total, low_noise, mid_noise, high_noise.
        """
        self.eval()
        batch_size = len(batch["inputs"])
        noise_buckets = ["low_noise", "mid_noise", "high_noise"]

        valid_gen_indices = [
            idx for idx, gen_pv in enumerate(batch["gen_pixel_values"]) if len(gen_pv) > 0
        ]
        num_gen_samples = len(valid_gen_indices)

        # Per-sample accumulators
        sample_bucket_losses = [[0.0 for _ in noise_buckets] for _ in range(batch_size)]
        sample_bucket_counts = [[0 for _ in noise_buckets] for _ in range(batch_size)]
        sample_total_losses = [0.0 for _ in range(batch_size)]

        if num_gen_samples == 0:
            return [{"total": 0.0, "low_noise": 0.0, "mid_noise": 0.0, "high_noise": 0.0} for _ in range(batch_size)]

        timesteps = torch.linspace(
            1 / num_valloss_timesteps,
            1 - 1 / num_valloss_timesteps,
            num_valloss_timesteps - 1,
        )

        for t_step in timesteps:
            t_val = t_step.item()
            if t_val < 1.0 / 3.0:
                bucket_idx = 0  # low noise
            elif t_val < 2.0 / 3.0:
                bucket_idx = 1  # mid noise
            else:
                bucket_idx = 2  # high noise

            with torch.no_grad():
                t = torch.full((num_gen_samples,), t_val, device=self.gen_model.device)
                t_expand_batch = t[:, None, None, None, None]

                targets = []
                noised_hidden_states = []
                source_hidden_states = []
                ref_hidden_states = []

                gen_idx = 0
                for idx, gen_pixel_values in enumerate(batch["gen_pixel_values"]):
                    if len(gen_pixel_values) == 0:
                        targets.append(None)
                        noised_hidden_states.append(None)
                        source_hidden_states.append(None)
                        ref_hidden_states.append(None)
                    else:
                        latents = self.get_latents(gen_pixel_values[0])
                        z_1 = torch.randn_like(latents)
                        eps = 1e-3
                        current_t_expand = t_expand_batch[gen_idx]
                        z_t = (1 - current_t_expand) * latents + (eps + (1 - eps) * current_t_expand) * z_1
                        target_velocity = (1 - eps) * z_1 - latents
                        z_t = z_t.to(latents.dtype)
                        target_velocity = target_velocity.to(latents.dtype)
                        targets.append(target_velocity)
                        noised_hidden_states.append(z_t)

                        source_pv = batch["source_pixel_values"][idx]
                        if len(source_pv) > 0:
                            source_hidden_states.append(self.get_latents(source_pv[0]))
                        else:
                            source_hidden_states.append(torch.zeros_like(latents))

                        ref_pvs = batch["ref_pixel_values"][idx]
                        if len(ref_pvs) > 0:
                            ref_hidden_states.append([self.get_latents(rpv) for rpv in ref_pvs])
                        else:
                            ref_hidden_states.append(None)
                        gen_idx += 1

                gen_timestep = (t * self.gen_scheduler.config.num_train_timesteps).to(self.gen_model.dtype)
                gen_encoder_hidden_states = self.fixed_t5_embeds.to(self.gen_model.device).expand(
                    num_gen_samples, -1, -1
                )

                _, pred = self.forward(
                    inputs=batch["inputs"],
                    gen_hidden_states=noised_hidden_states,
                    gen_timestep=gen_timestep,
                    gen_encoder_hidden_states=gen_encoder_hidden_states,
                    source_hidden_states=source_hidden_states,
                    source_scale=t,
                    ref_hidden_states=ref_hidden_states,
                )

                # Compute per-sample loss
                gen_count = 0
                for sample_idx, target in enumerate(targets):
                    if target is not None:
                        per_sample_loss = self.compute_gen_loss(
                            {"pred": pred[gen_count], "target": target}, mode="mean"
                        ).item()
                        sample_total_losses[sample_idx] += per_sample_loss
                        sample_bucket_losses[sample_idx][bucket_idx] += per_sample_loss
                        sample_bucket_counts[sample_idx][bucket_idx] += 1
                        gen_count += 1

        num_steps = num_valloss_timesteps - 1
        results = []
        for sample_idx in range(batch_size):
            result = {"total": sample_total_losses[sample_idx] / num_steps}
            for bi, bucket_name in enumerate(noise_buckets):
                count = sample_bucket_counts[sample_idx][bi]
                result[bucket_name] = (
                    sample_bucket_losses[sample_idx][bi] / count if count > 0 else 0.0
                )
            results.append(result)

        return results

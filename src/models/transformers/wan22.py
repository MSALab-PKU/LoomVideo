"""
Modified Wan2.2 Diffusion Transformer for LoomVideo.

Extends the HuggingFace Diffusers Wan transformer to support:
- Separate self-attention and cross-attention execution paths
- Variable-length sequence handling with cu_seq_lens
- Source video conditioning via learnable patch embedding
- Reference image/video conditioning with custom temporal RoPE offsets

Reference: https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_wan.py
"""

import math
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import register_to_config
from diffusers.utils import logging
from diffusers.models.attention import FeedForward
from diffusers.models.embeddings import get_1d_rotary_pos_embed
from diffusers.models.normalization import FP32LayerNorm
from diffusers.models.transformers.transformer_wan import (
    WanAttention,
    WanRotaryPosEmbed,
    WanTimeTextImageEmbedding,
    WanAttnProcessor,
    WanTransformerBlock,
    WanTransformer3DModel,
    _get_qkv_projections,
)

logger = logging.get_logger(__name__)


class WanAttnProcessor(WanAttnProcessor):
    """Attention processor with QKV extraction for cross-attention in LoomVideo."""

    _attention_backend = None

    def __init__(self):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("WanAttnProcessor requires PyTorch 2.0+.")

    def get_qkv(
        self,
        attn: "WanAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        """
        Extract Q, K, V projections with optional rotary embeddings.

        Returns:
            Tuple of (query, key, value, encoder_hidden_states, encoder_hidden_states_img).
        """
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            # 512 is the context length of the text encoder
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]

        query, key, value = _get_qkv_projections(
            attn, hidden_states, encoder_hidden_states
        )

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        if rotary_emb is not None:

            def apply_rotary_emb(
                hidden_states: torch.Tensor,
                freqs_cos: torch.Tensor,
                freqs_sin: torch.Tensor,
            ):
                x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
                cos = freqs_cos[..., 0::2]
                sin = freqs_sin[..., 1::2]
                out = torch.empty_like(hidden_states)
                out[..., 0::2] = x1 * cos - x2 * sin
                out[..., 1::2] = x1 * sin + x2 * cos
                return out.type_as(hidden_states)

            query = apply_rotary_emb(query, *rotary_emb)
            key = apply_rotary_emb(key, *rotary_emb)

        # [B, L, Num_heads, Head_dims] -> [B, Num_heads, L, Head_dims]
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        return query, key, value, encoder_hidden_states, encoder_hidden_states_img


class WanTransformerBlock(WanTransformerBlock):
    """
    Extended transformer block with separate self-attention and cross-attention paths.

    Adds methods for split execution:
    - forward_selfattn: self-attention only (for training with flash cross-attn)
    - forward_crossattn_later_layer: cross-attn + FFN (paired with forward_selfattn)
    """

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: Optional[int] = None,
    ):
        super().__init__(
            dim, ffn_dim, num_heads, qk_norm, cross_attn_norm, eps, added_kv_proj_dim
        )

        self.dim = dim
        self.num_heads = num_heads
        self.eps = eps

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            cross_attention_dim_head=None,
            processor=WanAttnProcessor(),
        )

        # 2. Cross-attention
        self.attn2 = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            added_kv_proj_dim=added_kv_proj_dim,
            cross_attention_dim_head=dim // num_heads,
            processor=WanAttnProcessor(),
        )
        self.norm2 = (
            FP32LayerNorm(dim, eps, elementwise_affine=True)
            if cross_attn_norm
            else nn.Identity()
        )

        # 3. Feed-forward
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def _parse_temb(self, temb: torch.Tensor):
        """Parse timestep embedding into shift/scale/gate components."""
        if temb.ndim == 4:
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table.unsqueeze(0) + temb.float()
            ).chunk(6, dim=2)
            shift_msa = shift_msa.squeeze(2)
            scale_msa = scale_msa.squeeze(2)
            gate_msa = gate_msa.squeeze(2)
            c_shift_msa = c_shift_msa.squeeze(2)
            c_scale_msa = c_scale_msa.squeeze(2)
            c_gate_msa = c_gate_msa.squeeze(2)
        else:
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table + temb.float()
            ).chunk(6, dim=1)
        return shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa

    def forward_selfattn(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
        cu_seq_lens: list = None,
    ) -> torch.Tensor:
        """
        Execute only the self-attention portion of this block.

        Args:
            hidden_states: Input tensor.
            temb: Timestep embedding (adaptive normalization parameters).
            rotary_emb: Rotary position embeddings.
            cu_seq_lens: Cumulative sequence lengths for per-sample attention.

        Returns:
            Hidden states after self-attention (before cross-attention and FFN).
        """
        shift_msa, scale_msa, gate_msa, _, _, _ = self._parse_temb(temb)

        norm_hidden_states = (self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)

        if cu_seq_lens is not None and len(cu_seq_lens) > 2:
            # Per-sample self-attention for variable-length sequences
            attn_outputs = []
            for i in range(len(cu_seq_lens) - 1):
                start = int(cu_seq_lens[i])
                end = int(cu_seq_lens[i + 1])
                sample_norm_hidden = norm_hidden_states[:, start:end, :]

                if isinstance(rotary_emb, (list, tuple)):
                    sample_rotary = [r[:, start:end, ...] if r is not None else None for r in rotary_emb]
                elif rotary_emb is not None:
                    sample_rotary = rotary_emb[:, start:end, ...]
                else:
                    sample_rotary = None
                sample_attn_output = self.attn1(sample_norm_hidden, None, None, sample_rotary)
                attn_outputs.append(sample_attn_output)

            attn_output = torch.cat(attn_outputs, dim=1)
        else:
            attn_output = self.attn1(norm_hidden_states, None, None, rotary_emb)

        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)

        return hidden_states

    def forward_crossattn_later_layer(
        self,
        hidden_states: torch.Tensor,
        attn_output: torch.Tensor,
        temb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Execute cross-attention residual addition and feed-forward network.

        Args:
            hidden_states: Current hidden states (after self-attention).
            attn_output: Cross-attention output to be added as residual.
            temb: Timestep embedding for adaptive normalization.

        Returns:
            Hidden states after cross-attention residual and FFN.
        """
        _, _, _, c_shift_msa, c_scale_msa, c_gate_msa = self._parse_temb(temb)

        hidden_states = hidden_states + attn_output

        # Feed-forward network
        norm_hidden_states = (self.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(hidden_states)
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)

        return hidden_states


class WanTransformer3DModel(WanTransformer3DModel):
    """
    Extended Wan 3D Transformer with source/reference conditioning support.

    Adds:
    - Source video conditioning via learnable patch embedding
    - Reference conditioning with custom temporal RoPE offsets
    - Separate early-layer and output-projection methods for flexible fusion
    """

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["patch_embedding", "condition_embedder", "norm"]
    _no_split_modules = ["WanTransformerBlock"]
    _keep_in_fp32_modules = [
        "time_embedder",
        "scale_shift_table",
        "norm1",
        "norm2",
        "norm3",
    ]
    _keys_to_ignore_on_load_unexpected = ["norm_added_q"]
    _repeated_blocks = ["WanTransformerBlock"]

    @register_to_config
    def __init__(
        self,
        patch_size: Tuple[int] = (1, 2, 2),
        num_attention_heads: int = 40,
        attention_head_dim: int = 128,
        in_channels: int = 16,
        out_channels: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 13824,
        num_layers: int = 40,
        cross_attn_norm: bool = True,
        qk_norm: Optional[str] = "rms_norm_across_heads",
        eps: float = 1e-6,
        image_dim: Optional[int] = None,
        added_kv_proj_dim: Optional[int] = None,
        rope_max_seq_len: int = 1024,
        pos_embed_seq_len: Optional[int] = None,
    ):
        super().__init__(
            patch_size,
            num_attention_heads,
            attention_head_dim,
            in_channels,
            out_channels,
            text_dim,
            freq_dim,
            ffn_dim,
            num_layers,
            cross_attn_norm,
            qk_norm,
            eps,
            image_dim,
            added_kv_proj_dim,
            rope_max_seq_len,
            pos_embed_seq_len,
        )

        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels
        self.in_channels = in_channels
        self.ffn_dim = ffn_dim
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.attention_head_dim = attention_head_dim
        self.num_attention_heads = num_attention_heads
        self.eps = eps
        self.added_kv_proj_dim = added_kv_proj_dim
        self.num_layers = num_layers
        self.inner_dim = inner_dim

        # 1. Patch & position embedding
        self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)
        self.patch_embedding = nn.Conv3d(
            in_channels, inner_dim, kernel_size=patch_size, stride=patch_size
        )

        # 2. Condition embeddings
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=text_dim,
            image_embed_dim=image_dim,
            pos_embed_seq_len=pos_embed_seq_len,
        )

        # 3. Transformer blocks
        self.blocks = nn.ModuleList(
            [
                WanTransformerBlock(
                    inner_dim,
                    ffn_dim,
                    num_attention_heads,
                    qk_norm,
                    cross_attn_norm,
                    eps,
                    added_kv_proj_dim,
                )
                for _ in range(num_layers)
            ]
        )

        # 4. Output norm & projection
        self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size))
        self.scale_shift_table = nn.Parameter(
            torch.randn(1, 2, inner_dim) / inner_dim**0.5
        )

        self.gradient_checkpointing = True

    def compute_ref_rotary_emb(
        self,
        ref_shape: Tuple[int, ...],
        time_offset: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute 3D RoPE for a reference latent with a custom temporal offset.

        Reference frames share the same temporal position (given by time_offset),
        while spatial dimensions use normal indices starting from 0.

        Args:
            ref_shape: (B, C, T, H, W) shape of the reference latent tensor.
            time_offset: Temporal position index for all frames (e.g., -10, -20).
            device: Target device.

        Returns:
            Tuple of (freqs_cos, freqs_sin) tensors.
        """
        _, _, num_frames, height, width = ref_shape
        p_t, p_h, p_w = self.config.patch_size
        ppf = num_frames // p_t
        pph = height // p_h
        ppw = width // p_w

        rope_module = self.rope
        freqs_dtype = torch.float64

        # Temporal: all frames share the same time_offset
        time_pos = np.array([time_offset] * ppf, dtype=np.float64)
        freq_cos_t, freq_sin_t = get_1d_rotary_pos_embed(
            rope_module.t_dim, time_pos, theta=10000.0,
            use_real=True, repeat_interleave_real=True, freqs_dtype=freqs_dtype,
        )

        # Spatial: normal indices [0, 1, ...]
        split_sizes = [rope_module.t_dim, rope_module.h_dim, rope_module.w_dim]
        precomputed_cos = rope_module.freqs_cos.split(split_sizes, dim=1)
        precomputed_sin = rope_module.freqs_sin.split(split_sizes, dim=1)

        freq_cos_h = precomputed_cos[1][:pph]
        freq_sin_h = precomputed_sin[1][:pph]
        freq_cos_w = precomputed_cos[2][:ppw]
        freq_sin_w = precomputed_sin[2][:ppw]

        # Broadcast to (ppf, pph, ppw, dim_*)
        freq_cos_t = freq_cos_t.to(device).view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freq_cos_h = freq_cos_h.to(device).view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freq_cos_w = freq_cos_w.to(device).view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)

        freq_sin_t = freq_sin_t.to(device).view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freq_sin_h = freq_sin_h.to(device).view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freq_sin_w = freq_sin_w.to(device).view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)

        freqs_cos = torch.cat([freq_cos_t, freq_cos_h, freq_cos_w], dim=-1).reshape(1, ppf * pph * ppw, 1, -1)
        freqs_sin = torch.cat([freq_sin_t, freq_sin_h, freq_sin_w], dim=-1).reshape(1, ppf * pph * ppw, 1, -1)

        return freqs_cos, freqs_sin

    def forward_early_layers(
        self,
        hidden_states: List[torch.Tensor],
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        source_hidden_states: Optional[List[torch.Tensor]] = None,
        source_scale: Optional[torch.Tensor] = None,
        ref_hidden_states: Optional[List[torch.Tensor]] = None,
    ):
        """
        Process patch embedding, timestep conditioning, source/ref concatenation.

        Prepares all inputs for the main transformer blocks by:
        1. Applying patch embedding to each latent sample
        2. Adding source conditioning (scaled by timestep)
        3. Concatenating reference latents with custom temporal RoPE
        4. Computing timestep projections

        Args:
            hidden_states: List of latent tensors (None for non-generation samples).
            timestep: Diffusion timestep for each generation sample.
            encoder_hidden_states: T5 text encoder hidden states.
            source_hidden_states: Optional source video latents.
            source_scale: Timestep-dependent scale for source conditioning.
            ref_hidden_states: Optional list of reference latent lists.

        Returns:
            Tuple of processed tensors and metadata for the transformer blocks.
        """
        hidden_states_list = []
        shape_list = []
        rotary_emb_cos = []
        rotary_emb_sin = []
        cu_seq_lens = [0]
        sample_index = []
        seq_lens = []
        gen_seq_lens = []

        for index, hidden_state in enumerate(hidden_states):
            if hidden_state is not None:
                valid_sample_idx = len(seq_lens)
                shape = hidden_state.shape

                hidden_state = self.patch_embedding(hidden_state)
                hidden_state = hidden_state.flatten(2).transpose(1, 2)  # [1, L, C]

                # Add source video conditioning
                if (source_hidden_states is not None
                        and source_hidden_states[index] is not None
                        and source_scale is not None
                        and hasattr(self, 'source_patch_embedding')):
                    source_input = source_hidden_states[index]
                    source_encoded = self.source_patch_embedding(source_input)
                    source_encoded = source_encoded.flatten(2).transpose(1, 2)
                    source_scale_idx = index if source_scale.shape[0] == len(hidden_states) else valid_sample_idx
                    hidden_state = hidden_state + source_encoded * source_scale[source_scale_idx]

                # Record original sequence length (for loss computation, excluding ref tokens)
                original_seq_len = hidden_state.shape[1]
                gen_seq_lens.append(original_seq_len)

                # Compute rotary embeddings for the main latent
                fake_tensor = torch.empty(shape, device=hidden_state.device, dtype=hidden_state.dtype)
                rotary_emb = self.rope(fake_tensor)

                # Concatenate reference latents with negative temporal offsets
                if (ref_hidden_states is not None and ref_hidden_states[index] is not None):
                    ref_list = ref_hidden_states[index]
                    for ref_idx, ref_input in enumerate(ref_list):
                        ref_encoded = self.patch_embedding(ref_input)
                        ref_encoded = ref_encoded.flatten(2).transpose(1, 2)
                        hidden_state = torch.cat([hidden_state, ref_encoded], dim=1)

                        # Negative temporal offset: -10, -20, -30, ... for each ref
                        ref_time_offset = -10 * (ref_idx + 1)
                        ref_rotary_emb = self.compute_ref_rotary_emb(
                            ref_shape=ref_input.shape,
                            time_offset=ref_time_offset,
                            device=hidden_state.device,
                        )
                        rotary_emb = (
                            torch.cat([rotary_emb[0], ref_rotary_emb[0]], dim=1),
                            torch.cat([rotary_emb[1], ref_rotary_emb[1]], dim=1),
                        )

                hidden_states_list.append(hidden_state)
                rotary_emb_cos.append(rotary_emb[0])
                rotary_emb_sin.append(rotary_emb[1])
                cu_seq_lens.append(cu_seq_lens[-1] + hidden_state.shape[1])
                sample_index.append(index)
                shape_list.append(shape)
                seq_lens.append(hidden_state.shape[1])

        hidden_states = torch.cat(hidden_states_list, dim=1)  # [1, sum(L), C]
        rotary_emb_cos = torch.cat(rotary_emb_cos, dim=1)
        rotary_emb_sin = torch.cat(rotary_emb_sin, dim=1)
        rotary_emb = (rotary_emb_cos, rotary_emb_sin)

        # Timestep conditioning
        if timestep.ndim == 2:
            ts_seq_len = timestep.shape[1]
            timestep = timestep.flatten()
        else:
            ts_seq_len = None

        temb, timestep_proj, encoder_hidden_states, _ = (
            self.condition_embedder(
                timestep,
                encoder_hidden_states,
                timestep_seq_len=ts_seq_len,
            )
        )
        if ts_seq_len is not None:
            timestep_proj = timestep_proj.unflatten(2, (6, -1))
        else:
            timestep_proj = timestep_proj.unflatten(1, (6, -1))

        # Expand timestep projections to match sequence lengths (zero for ref tokens)
        num_valid_samples = len(seq_lens)
        if timestep_proj.shape[0] == num_valid_samples and num_valid_samples > 0:
            expanded_timestep_proj = []
            expanded_temb = []
            for i, length in enumerate(seq_lens):
                gen_len = gen_seq_lens[i]
                ref_len = length - gen_len
                if ts_seq_len is None:
                    expanded_timestep_proj.append(timestep_proj[i:i + 1].expand(gen_len, -1, -1))
                    if temb is not None:
                        expanded_temb.append(temb[i:i + 1].expand(gen_len, -1))
                    # Zero timestep embedding for ref tokens
                    if ref_len > 0:
                        expanded_timestep_proj.append(
                            torch.zeros(ref_len, timestep_proj.shape[1], timestep_proj.shape[2],
                                        device=timestep_proj.device, dtype=timestep_proj.dtype)
                        )
                        if temb is not None:
                            expanded_temb.append(
                                torch.zeros(ref_len, temb.shape[1], device=temb.device, dtype=temb.dtype)
                            )
                else:
                    expanded_timestep_proj.append(timestep_proj[i])
                    if temb is not None:
                        expanded_temb.append(temb[i])

            if ts_seq_len is None:
                timestep_proj = torch.cat(expanded_timestep_proj, dim=0).unsqueeze(0)
                if temb is not None:
                    temb = torch.cat(expanded_temb, dim=0).unsqueeze(0)
            else:
                timestep_proj = torch.cat(expanded_timestep_proj, dim=0).unsqueeze(0)
                if temb is not None:
                    temb = torch.cat(expanded_temb, dim=0).unsqueeze(0)

        return (
            hidden_states,
            encoder_hidden_states,
            timestep_proj,
            rotary_emb,
            temb,
            shape_list,
            cu_seq_lens,
            sample_index,
            gen_seq_lens,
        )

    def get_output(self, hidden_states: torch.Tensor, temb: torch.Tensor, cu_seq_lens, shape, gen_seq_lens=None):
        """
        Project hidden states back to pixel space via unpatchify.

        Args:
            hidden_states: Transformer output tensor.
            temb: Timestep embedding for final adaptive normalization.
            cu_seq_lens: Cumulative sequence lengths.
            shape: Original latent shapes per sample.
            gen_seq_lens: Original sequence lengths (excluding ref tokens).
                If provided, only these tokens are projected (ref tokens discarded).

        Returns:
            List of output tensors, one per sample.
        """
        output_list = []

        for index in range(len(cu_seq_lens) - 1):
            start_idx = cu_seq_lens[index]
            # Only project original tokens (exclude ref tokens)
            if gen_seq_lens is not None:
                end_idx = cu_seq_lens[index] + gen_seq_lens[index]
            else:
                end_idx = cu_seq_lens[index + 1]
            hidden_state = hidden_states[:, start_idx:end_idx, :]

            if temb.ndim == 3 and temb.shape[1] == hidden_states.shape[1]:
                sample_temb = temb[:, start_idx:end_idx, :]
            elif temb.ndim == 2 and temb.shape[0] > 1:
                sample_temb = temb[index:index + 1]
            else:
                sample_temb = temb

            batch_size, num_channels, num_frames, height, width = shape[index]
            p_t, p_h, p_w = self.config.patch_size
            post_patch_num_frames = num_frames // p_t
            post_patch_height = height // p_h
            post_patch_width = width // p_w

            # Adaptive normalization for output
            if sample_temb.ndim == 3:
                shift, scale = (
                    self.scale_shift_table.unsqueeze(0) + sample_temb.unsqueeze(2)
                ).chunk(2, dim=2)
                shift = shift.squeeze(2)
                scale = scale.squeeze(2)
            else:
                shift, scale = (self.scale_shift_table + sample_temb.unsqueeze(1)).chunk(2, dim=1)

            shift = shift.to(hidden_state.device)
            scale = scale.to(hidden_state.device)

            hidden_state = (
                self.norm_out(hidden_state.float()) * (1 + scale) + shift
            ).type_as(hidden_state)
            hidden_state = self.proj_out(hidden_state)

            # Unpatchify: reshape back to video dimensions
            hidden_state = hidden_state.reshape(
                batch_size,
                post_patch_num_frames,
                post_patch_height,
                post_patch_width,
                p_t,
                p_h,
                p_w,
                -1,
            )
            hidden_state = hidden_state.permute(0, 7, 1, 4, 2, 5, 3, 6)
            output = hidden_state.flatten(6, 7).flatten(4, 5).flatten(2, 3)
            output_list.append(output)

        return output_list

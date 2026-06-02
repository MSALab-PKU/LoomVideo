"""
Model utilities for LoomVideo.

Provides:
- unfreeze_model: selective parameter unfreezing based on config
- load_model: build LoomVideo model from config
- load_checkpoint: load trained weights with EMA priority
"""

import logging
import os
from collections.abc import Mapping, Sequence

import torch

logger = logging.getLogger(__name__)


def check_match(name: str, config_node) -> bool:
    """
    Recursively check whether a parameter name matches the unfreeze config.

    Args:
        name: Full parameter name (e.g., 'gen_model.blocks.0.attn1.to_q.weight').
        config_node: Config node specifying which params to unfreeze.
            - "all" or True: match everything
            - dict: check key substring in name, recurse on value
            - list: match if any item is a substring of name
            - str: match if substring of name
    """
    if config_node == "all" or config_node is True:
        return True

    if not config_node:
        return False

    if isinstance(config_node, Mapping):
        for key, value in config_node.items():
            if key in name:
                if check_match(name, value):
                    return True
        return False

    if isinstance(config_node, Sequence) and not isinstance(config_node, str):
        for item in config_node:
            if item == "all":
                return True
            if item in name:
                return True
        return False

    if isinstance(config_node, str):
        return config_node in name

    return False


def unfreeze_model(model, model_config):
    """
    Selectively unfreeze model parameters based on config.

    Args:
        model: The model whose parameters to selectively unfreeze.
        model_config: Config node specifying trainable modules.
    """
    if not model_config:
        return

    for name, param in model.named_parameters():
        if check_match(name, model_config):
            param.requires_grad = True


def load_model(config):
    """
    Build LoomVideo model from config.

    Args:
        config: OmegaConf config object with model.und and model.gen fields.

    Returns:
        LoomVideo model instance.
    """
    from .transformers.loomvideo import LoomVideo
    return LoomVideo(config)


def load_checkpoint(model, ckpt_path: str):
    """
    Load trained weights into model with EMA priority.

    Loading priority:
        1. ckpt_path/ema/ema.pth — EMA shadow params (best generation quality)
        2. ckpt_path/latest/<component>.pth — per-component state dicts
        3. ckpt_path as a single file (.safetensors or .pt) — full state dict

    Args:
        model: LoomVideo model to load weights into.
        ckpt_path: Path to checkpoint directory or single file.

    Raises:
        FileNotFoundError: If ckpt_path does not exist.
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Case 1: ckpt_path is a single file (safetensors or .pt/.pth)
    if os.path.isfile(ckpt_path):
        logger.info(f"Loading checkpoint from file: {ckpt_path}")
        if ckpt_path.endswith(".safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(ckpt_path, device="cpu")
        else:
            state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning(f"Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
        if unexpected:
            logger.warning(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
        logger.info("Checkpoint loaded successfully.")
        return

    # Case 2: ckpt_path is a directory
    ema_dir = os.path.join(ckpt_path, "ema")
    latest_dir = os.path.join(ckpt_path, "latest")

    # Priority 1: EMA weights
    ema_file = os.path.join(ema_dir, "ema.pth")
    if os.path.isdir(ema_dir) and os.path.exists(ema_file):
        logger.info(f"Loading EMA weights from: {ema_file}")
        ema_state = torch.load(ema_file, map_location="cpu", weights_only=True)
        shadow_params = ema_state["shadow_params"]
        trainable_params = list(model.get_trainable_parameters())

        if len(shadow_params) != len(trainable_params):
            logger.warning(
                f"EMA param count ({len(shadow_params)}) != "
                f"trainable param count ({len(trainable_params)}). "
                f"Attempting partial load..."
            )

        with torch.no_grad():
            for param, ema_param in zip(trainable_params, shadow_params):
                param.copy_(ema_param.to(param.device))

        logger.info(f"Loaded EMA weights ({len(shadow_params)} params).")
        return

    # Priority 2: Per-component state dicts in latest/ subdirectory
    # Priority 3: Per-component state dicts directly in ckpt_path/
    for search_dir in [latest_dir, ckpt_path]:
        if not os.path.isdir(search_dir):
            continue
        components = model.get_trainable_components()
        loaded_any = False

        for name, component in components.items():
            pth_file = os.path.join(search_dir, f"{name}.pth")
            if os.path.exists(pth_file):
                if not loaded_any:
                    logger.info(f"Loading per-component weights from: {search_dir}")
                state_dict = torch.load(pth_file, map_location="cpu", weights_only=True)
                missing, unexpected = component.load_state_dict(state_dict, strict=False)
                logger.info(f"Loaded {name}: missing={len(missing)}, unexpected={len(unexpected)}")
                loaded_any = True

        if loaded_any:
            return

    raise FileNotFoundError(
        f"No valid checkpoint found under {ckpt_path}. "
        f"Expected: ema/ema.pth, latest/<component>.pth, "
        f"<component>.pth, or a single .safetensors/.pt file."
    )

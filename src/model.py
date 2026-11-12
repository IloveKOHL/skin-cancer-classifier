"""Model creation and checkpoint loading (timm backbones)."""

from __future__ import annotations

from pathlib import Path

import timm
import torch
from torch import nn


def build_model(cfg: dict, num_classes: int, pretrained: bool | None = None) -> nn.Module:
    """Create a timm model with a fresh classification head.

    Swap the backbone by changing ``model.name`` in the config, e.g.
    ``efficientnet_b0``, ``resnet50`` or ``convnext_tiny``.
    """
    if pretrained is None:
        pretrained = cfg["model"]["pretrained"]
    return timm.create_model(
        cfg["model"]["name"],
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=cfg["model"]["drop_rate"],
        # .get: checkpoints written before this option existed still load
        drop_path_rate=cfg["model"].get("drop_path_rate", 0.0),
    )


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[nn.Module, dict]:
    """Load a checkpoint written by ``src.train``.

    Returns the model in eval mode and the checkpoint dict (config, class names, ...).
    """
    # Checkpoints only hold tensors and plain Python containers, so the safe
    # weights_only loader is enough.
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = build_model(ckpt["config"], num_classes=len(ckpt["class_names"]), pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, ckpt

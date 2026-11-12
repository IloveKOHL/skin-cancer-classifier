"""Predict the class of one or more images with a trained checkpoint.

Usage:
    python -m src.predict --checkpoint runs/<run>/best.pt path/to/image.jpg [more.jpg ...]

Research/learning tool only - the output is NOT a medical diagnosis.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from PIL import Image
from torch import nn

from src.dataset import build_transforms
from src.model import load_checkpoint
from src.utils import get_device


@torch.no_grad()
def predict_image(
    model: nn.Module, image: Image.Image, transform, device: torch.device
) -> np.ndarray:
    """Return the class probabilities for a single PIL image."""
    model.eval()
    batch = transform(image.convert("RGB")).unsqueeze(0).to(device)
    return torch.softmax(model(batch).float(), dim=1)[0].cpu().numpy()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("images", nargs="+", help="Image file(s)")
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    device = get_device(args.device)
    model, ckpt = load_checkpoint(args.checkpoint, device)
    transform = build_transforms(ckpt["config"], train=False)
    class_names = ckpt["class_names"]

    for path in args.images:
        probs = predict_image(model, Image.open(path), transform, device)
        ranked = sorted(zip(class_names, probs), key=lambda item: -item[1])
        print(f"{path}: " + ", ".join(f"{name}={p:.3f}" for name, p in ranked))
    print("\nNot a medical device. This output is not a diagnosis.")


if __name__ == "__main__":
    main()

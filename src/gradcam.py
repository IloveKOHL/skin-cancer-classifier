"""Grad-CAM heatmaps: which image regions pushed the model towards a class?

Usage:
    # heatmaps for random test-split images
    python -m src.gradcam --checkpoint runs/<run>/best.pt --num-samples 8
    # heatmaps for your own files
    python -m src.gradcam --checkpoint runs/<run>/best.pt --images a.jpg b.jpg

Grad-CAM (Selvaraju et al., 2017) weights the last convolutional feature maps by
the gradient of the class score. It is a rough sanity check (does the model look
at the lesion or at rulers/hair/vignettes?), not a proof of correct reasoning.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from src.dataset import build_transforms, prepare_data
from src.model import load_checkpoint
from src.utils import get_device


def grad_cam(
    model: nn.Module, batch: torch.Tensor, target_class: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Grad-CAM for a single preprocessed image (shape 1x3xHxW).

    Uses timm's ``forward_features`` / ``forward_head`` split, so it works for any
    timm CNN whose feature output is a (N, C, H, W) map (EfficientNet, ResNet,
    ConvNeXt, ...) without model-specific layer names.

    Returns:
        cam: heatmap in [0, 1] with the spatial size of the input image.
        probs: class probabilities.
    """
    model.eval()
    with torch.enable_grad():
        features = model.forward_features(batch)
        if features.ndim != 4:
            raise ValueError("Grad-CAM needs a CNN with (N, C, H, W) feature maps.")
        features.retain_grad()
        logits = model.forward_head(features)
        probs = torch.softmax(logits, dim=1)[0]
        if target_class is None:
            target_class = int(probs.argmax())
        model.zero_grad(set_to_none=True)
        logits[0, target_class].backward()

    weights = features.grad.mean(dim=(2, 3), keepdim=True)  # importance of each channel
    cam = F.relu((weights * features).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=batch.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam - cam.min()
    cam = cam / cam.max().clamp_min(1e-8)
    return cam.detach().cpu().numpy(), probs.detach().cpu().numpy()


def overlay_heatmap(image: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Blend a [0, 1] heatmap over the image using the 'jet' colour map."""
    import matplotlib

    heat = matplotlib.colormaps["jet"](cam)[..., :3]  # RGBA -> RGB in [0, 1]
    heat_img = Image.fromarray((heat * 255).astype(np.uint8)).resize(image.size, Image.BILINEAR)
    return Image.blend(image.convert("RGB"), heat_img, alpha)


def explain_image(
    model: nn.Module, image: Image.Image, transform, device: torch.device,
    target_class: int | None = None,
) -> tuple[Image.Image, np.ndarray]:
    """Full pipeline for one PIL image: returns (overlay image, probabilities)."""
    image = image.convert("RGB")
    batch = transform(image).unsqueeze(0).to(device)
    cam, probs = grad_cam(model, batch, target_class)
    return overlay_heatmap(image, cam), probs


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--images", nargs="*", default=[], help="Image files (optional)")
    parser.add_argument("--num-samples", type=int, default=8,
                        help="Random test images to use when --images is not given")
    parser.add_argument("--output", default=None, help="Output folder (default: <run>/gradcam)")
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    device = get_device(args.device)
    model, ckpt = load_checkpoint(args.checkpoint, device)
    cfg, class_names = ckpt["config"], ckpt["class_names"]
    transform = build_transforms(cfg, train=False)

    if args.images:
        items = [(Path(p), None) for p in args.images]
    else:
        test = prepare_data(cfg)[0]["test"]
        sample = test.sample(n=min(args.num_samples, len(test)), random_state=cfg["seed"])
        items = [(Path(row.path), class_names[row.label]) for row in sample.itertuples()]

    out_dir = Path(args.output) if args.output else Path(args.checkpoint).parent / "gradcam"
    out_dir.mkdir(parents=True, exist_ok=True)
    for path, true_name in items:
        image = Image.open(path).convert("RGB")
        overlay, probs = explain_image(model, image, transform, device)
        pred = int(probs.argmax())

        # Original and overlay side by side.
        canvas = Image.new("RGB", (image.width * 2, image.height))
        canvas.paste(image, (0, 0))
        canvas.paste(overlay, (image.width, 0))
        out_path = out_dir / f"{path.stem}_pred-{class_names[pred]}.png"
        canvas.save(out_path)
        truth = f", true={true_name}" if true_name else ""
        print(f"{path.name}: pred={class_names[pred]} ({probs[pred]:.3f}){truth} -> {out_path}")
    return out_dir


if __name__ == "__main__":
    main()

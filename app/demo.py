"""Gradio demo: upload a dermatoscopic image, get prediction + Grad-CAM.

Usage:
    python app/demo.py --checkpoint runs/<run>/best.pt

NOT a medical device. For learning and research only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as a plain script from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gradio as gr  # noqa: E402

from src.dataset import build_transforms  # noqa: E402
from src.gradcam import explain_image  # noqa: E402
from src.model import load_checkpoint  # noqa: E402
from src.utils import get_device  # noqa: E402

DISCLAIMER = """
## ⚠️ Not a medical device - research and education only

This demo is a student learning project. It **cannot diagnose skin cancer** and is
**not a substitute for a dermatologist**. The model was trained on a small public
dataset of dermatoscopic images (mostly light skin) and will be wrong on many
images - especially ordinary phone photos. **If a skin lesion worries you, see a doctor.**
"""


def build_demo(checkpoint: str, device_name: str | None = None) -> gr.Blocks:
    device = get_device(device_name)
    model, ckpt = load_checkpoint(checkpoint, device)
    class_names = ckpt["class_names"]
    transform = build_transforms(ckpt["config"], train=False)

    def predict(image):
        if image is None:
            return None, None
        overlay, probs = explain_image(model, image, transform, device)
        return {name: float(p) for name, p in zip(class_names, probs)}, overlay

    with gr.Blocks(title="Skin lesion classifier (research demo)") as demo:
        gr.Markdown("# Skin lesion classifier - research demo")
        gr.Markdown(DISCLAIMER)
        with gr.Row():
            with gr.Column():
                image_in = gr.Image(type="pil", label="Dermatoscopic image")
                button = gr.Button("Classify", variant="primary")
            with gr.Column():
                label_out = gr.Label(num_top_classes=len(class_names), label="Model output")
                cam_out = gr.Image(label="Grad-CAM (regions that influenced the prediction)")
        gr.Markdown(
            f"Model: `{ckpt['config']['model']['name']}` · mode: "
            f"`{ckpt['config']['task']['mode']}` · classes: {', '.join(class_names)}. "
            "Probabilities are uncalibrated model scores, not clinical risk estimates."
        )
        button.click(predict, inputs=image_in, outputs=[label_out, cam_out])
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt")
    parser.add_argument("--device", default=None)
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    # Local only: share=False keeps the demo off the public internet.
    build_demo(args.checkpoint, args.device).launch(server_port=args.port, share=False)


if __name__ == "__main__":
    main()

"""Gradio demo: upload a dermatoscopic image, get prediction + Grad-CAM.

Usage:
    python app/demo.py --checkpoint runs/<run>/best.pt
    # compare several models side by side on the same image
    python app/demo.py --checkpoint runs/<binary_run>/best.pt runs/<multiclass_run>/best.pt

NOT a medical device. For learning and research only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as a plain script from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gradio as gr  # noqa: E402

from src.dataset import DISPLAY_NAMES, MALIGNANT, build_transforms  # noqa: E402
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


def make_predict_fn(model, class_names: list[str], transform, device):
    """Build the callback: PIL image -> (summary markdown, label scores, Grad-CAM overlay)."""

    def predict(image):
        if image is None:
            return "", None, None
        overlay, probs = explain_image(model, image, transform, device)
        top = int(probs.argmax())
        summary = (
            f"### Model output: {DISPLAY_NAMES[class_names[top]]}\n"
            f"Model score for this class: {probs[top]:.0%}"
        )
        if len(class_names) > 2:
            # Multiclass: also show how much score the malignant group gets in total.
            group = sum(float(p) for name, p in zip(class_names, probs) if name in MALIGNANT)
            summary += (
                f" · combined score of the malignant / pre-cancerous classes "
                f"(mel, bcc, akiec): {group:.0%}"
            )
        summary += (
            "\n\nThis is the output of an experimental model, **not a diagnosis**. "
            "Scores are not risk estimates and are often over-confident."
        )
        scores = {DISPLAY_NAMES[name]: float(p) for name, p in zip(class_names, probs)}
        return summary, scores, overlay

    return predict


def build_demo(checkpoints: str | list[str], device_name: str | None = None) -> gr.Blocks:
    """One result column per checkpoint; all models see the same uploaded image."""
    if isinstance(checkpoints, (str, Path)):
        checkpoints = [checkpoints]
    device = get_device(device_name)

    models = []
    for path in checkpoints:
        model, ckpt = load_checkpoint(path, device)
        cfg, class_names = ckpt["config"], ckpt["class_names"]
        models.append(
            {
                "title": f"{cfg['task']['mode'].capitalize()} model",
                "info": f"`{cfg['model']['name']}` · run `{Path(path).parent.name}`",
                "num_classes": len(class_names),
                "predict": make_predict_fn(
                    model, class_names, build_transforms(cfg, train=False), device
                ),
            }
        )

    def predict_all(image):
        results = []
        for entry in models:
            results.extend(entry["predict"](image))
        return results

    with gr.Blocks(title="Skin lesion classifier (research demo)") as demo:
        gr.Markdown("# Skin lesion classifier - research demo")
        gr.Markdown(DISCLAIMER)
        with gr.Row():
            with gr.Column(scale=1):
                image_in = gr.Image(type="pil", label="Dermatoscopic image")
                button = gr.Button("Classify", variant="primary")
            outputs = []
            for entry in models:
                with gr.Column(scale=1):
                    gr.Markdown(f"## {entry['title']}\n{entry['info']}")
                    outputs += [
                        gr.Markdown(),
                        gr.Label(num_top_classes=entry["num_classes"], label="Scores per class"),
                        gr.Image(label="Grad-CAM (red = regions that influenced the prediction)"),
                    ]
        gr.Markdown(
            "Probabilities are uncalibrated model scores, not clinical risk estimates. "
            "Different models can disagree on the same image."
        )
        button.click(predict_all, inputs=image_in, outputs=outputs)
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--checkpoint", required=True, nargs="+", help="One or more best.pt files to compare"
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    # Local only: share=False keeps the demo off the public internet.
    build_demo(args.checkpoint, args.device).launch(server_port=args.port, share=False)


if __name__ == "__main__":
    main()

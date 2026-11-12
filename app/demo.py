"""Gradio demo: upload a dermatoscopic image, get prediction + Grad-CAM.

Usage:
    python app/demo.py --checkpoint runs/<run>/best.pt
    # compare several models side by side on the same image, German interface
    python app/demo.py --checkpoint runs/<binary_run>/best.pt runs/<multiclass_run>/best.pt \
        --language de

NOT a medical device. For learning and research only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as a plain script from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gradio as gr  # noqa: E402
from texts import CLASSES, UI, glossary_markdown  # noqa: E402

from src.dataset import MALIGNANT, build_transforms  # noqa: E402
from src.gradcam import explain_image  # noqa: E402
from src.model import load_checkpoint  # noqa: E402
from src.utils import get_device  # noqa: E402


def make_predict_fn(model, class_names: list[str], transform, device, language: str = "en"):
    """Build the callback: PIL image -> (summary markdown, label scores, Grad-CAM overlay)."""
    ui, classes = UI[language], CLASSES[language]

    def predict(image):
        if image is None:
            return "", None, None
        overlay, probs = explain_image(model, image, transform, device)
        top = int(probs.argmax())
        name, explanation = classes[class_names[top]]
        lines = [
            f"### {ui['result']}: {name}",
            f"**{ui['meaning']}:** {explanation}",
            f"**{ui['score']}:** {probs[top]:.0%} ({ui['score_help']})",
        ]
        if len(class_names) > 2:
            # Multiclass: also show how much score the malignant group gets in total.
            group = sum(float(p) for code, p in zip(class_names, probs) if code in MALIGNANT)
            lines.append(f"**{ui['group']}:** {group:.0%}")
        lines.append(ui["not_diagnosis"])
        scores = {classes[code][0]: float(p) for code, p in zip(class_names, probs)}
        return "\n\n".join(lines), scores, overlay

    return predict


def build_demo(
    checkpoints: str | list[str], device_name: str | None = None, language: str = "en"
) -> gr.Blocks:
    """One result column per checkpoint; all models see the same uploaded image."""
    if isinstance(checkpoints, (str, Path)):
        checkpoints = [checkpoints]
    device = get_device(device_name)
    ui = UI[language]

    models, glossary_classes = [], []
    for path in checkpoints:
        model, ckpt = load_checkpoint(path, device)
        cfg, class_names = ckpt["config"], ckpt["class_names"]
        glossary_classes += [name for name in class_names if name not in glossary_classes]
        models.append(
            {
                "title": ui[f"model_{cfg['task']['mode']}"],
                "info": f"`{cfg['model']['name']}` · run `{Path(path).parent.name}`",
                "num_classes": len(class_names),
                "predict": make_predict_fn(
                    model, class_names, build_transforms(cfg, train=False), device, language
                ),
            }
        )

    def predict_all(image):
        results = []
        for entry in models:
            results.extend(entry["predict"](image))
        return results

    with gr.Blocks(title=ui["page_title"]) as demo:
        gr.Markdown(ui["title"])
        gr.Markdown(ui["disclaimer"])
        with gr.Row():
            with gr.Column(scale=1):
                image_in = gr.Image(type="pil", label=ui["image"])
                button = gr.Button(ui["button"], variant="primary")
            outputs = []
            for entry in models:
                with gr.Column(scale=1):
                    gr.Markdown(f"## {entry['title']}\n{entry['info']}")
                    outputs += [
                        gr.Markdown(),
                        gr.Label(num_top_classes=entry["num_classes"], label=ui["scores"]),
                        gr.Image(label=ui["gradcam"]),
                    ]
        with gr.Accordion(ui["glossary"], open=True):
            gr.Markdown(glossary_markdown(language, glossary_classes))
        gr.Markdown(ui["footer"])
        button.click(predict_all, inputs=image_in, outputs=outputs)
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--checkpoint", required=True, nargs="+", help="One or more best.pt files to compare"
    )
    parser.add_argument("--language", default="en", choices=sorted(UI), help="Interface language")
    parser.add_argument("--device", default=None)
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    # Local only: share=False keeps the demo off the public internet.
    build_demo(args.checkpoint, args.device, args.language).launch(
        server_port=args.port, share=False
    )


if __name__ == "__main__":
    main()

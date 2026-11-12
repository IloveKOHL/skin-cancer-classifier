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
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from texts import CLASSES, UI, glossary_markdown  # noqa: E402

from src.dataset import MALIGNANT, build_transforms  # noqa: E402
from src.gradcam import explain_image, grad_cam, overlay_heatmap  # noqa: E402
from src.model import load_checkpoint  # noqa: E402
from src.utils import get_device  # noqa: E402


def summarise(probs, class_names: list[str], language: str) -> tuple[str, dict[str, float]]:
    """Plain-language summary (markdown) and per-class scores for the label widget."""
    ui, classes = UI[language], CLASSES[language]
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
    return "\n\n".join(lines), scores


def make_predict_fn(model, class_names: list[str], transform, device, language: str = "en"):
    """Build the callback: PIL image -> (summary markdown, label scores, Grad-CAM overlay)."""

    def predict(image):
        if image is None:
            return "", None, None
        overlay, probs = explain_image(model, image, transform, device)
        return (*summarise(probs, class_names, language), overlay)

    return predict


def make_ensemble_predict_fn(members, class_names: list[str], device, language: str = "en"):
    """Like ``make_predict_fn`` but averages several models.

    ``members`` is a list of ``(model, transform)``. Probabilities are averaged over
    all models and over four flipped copies of the image (test-time augmentation),
    exactly like ``src.evaluate --tta`` with several checkpoints. The heat map is
    the mean of the members' Grad-CAM maps for the ensemble's predicted class.
    """

    def predict(image):
        if image is None:
            return "", None, None
        image = image.convert("RGB")
        batches = [transform(image).unsqueeze(0).to(device) for _, transform in members]

        probs = 0.0
        with torch.no_grad():
            for (model, _), batch in zip(members, batches):
                for dims in [(), (3,), (2,), (2, 3)]:
                    view = torch.flip(batch, dims) if dims else batch
                    probs = probs + torch.softmax(model(view).float(), dim=1)[0]
        probs = (probs / (4 * len(members))).cpu().numpy()

        top = int(probs.argmax())
        cams = []
        for (model, _), batch in zip(members, batches):
            cam, _ = grad_cam(model, batch, target_class=top)
            cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize(image.size)
            cams.append(np.asarray(cam_img, dtype=np.float32) / 255.0)
        overlay = overlay_heatmap(image, np.mean(cams, axis=0))
        return (*summarise(probs, class_names, language), overlay)

    return predict


def build_demo(
    checkpoints: str | list[str],
    device_name: str | None = None,
    language: str = "en",
    ensemble: bool = False,
) -> gr.Blocks:
    """One result column per checkpoint; all models see the same uploaded image.

    With ``ensemble=True`` the checkpoints are combined into a single averaged model.
    """
    if isinstance(checkpoints, (str, Path)):
        checkpoints = [checkpoints]
    device = get_device(device_name)
    ui = UI[language]

    models, glossary_classes, members = [], [], []
    for path in checkpoints:
        model, ckpt = load_checkpoint(path, device)
        cfg, class_names = ckpt["config"], ckpt["class_names"]
        if ensemble:
            if glossary_classes and class_names != glossary_classes:
                raise ValueError("All checkpoints of an ensemble must have the same classes.")
            glossary_classes = class_names
            members.append((model, build_transforms(cfg, train=False)))
            continue
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

    if ensemble:
        models.append(
            {
                "title": ui["model_ensemble"].format(n=len(members)),
                "info": " · ".join(f"`{Path(p).parent.name}`" for p in checkpoints),
                "num_classes": len(glossary_classes),
                "predict": make_ensemble_predict_fn(members, glossary_classes, device, language),
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
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help="Combine all checkpoints into one averaged model instead of showing them side by side",
    )
    parser.add_argument("--language", default="en", choices=sorted(UI), help="Interface language")
    parser.add_argument("--device", default=None)
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    # Local only: share=False keeps the demo off the public internet.
    build_demo(args.checkpoint, args.device, args.language, args.ensemble).launch(
        server_port=args.port, share=False
    )


if __name__ == "__main__":
    main()

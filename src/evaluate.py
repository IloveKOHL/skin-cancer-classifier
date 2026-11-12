"""Evaluate a trained checkpoint on the val or test split.

Usage:
    python -m src.evaluate --checkpoint runs/<run>/best.pt --split test

Writes metrics.json, confusion_matrix.png and classification_report.txt to
``<run>/eval_<split>/``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report
from torch import nn
from torch.utils.data import DataLoader

from src.dataset import build_dataloaders, prepare_data
from src.model import load_checkpoint
from src.utils import (
    compute_metrics,
    flag_metrics,
    get_device,
    save_confusion_matrix,
    threshold_for_sensitivity,
)


def positive_class(class_names: list[str], mode: str) -> tuple[int, str]:
    """The class whose detection matters most: malignant (binary) or melanoma."""
    name = "malignant" if mode == "binary" else "mel"
    return class_names.index(name), name


@torch.no_grad()
def predict_loader(
    model: nn.Module, loader: DataLoader, device: torch.device, amp: bool = False
) -> tuple[float, np.ndarray, np.ndarray]:
    """Run the model over a loader. Returns (mean loss, labels, class probabilities)."""
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss, labels, probs = 0.0, [], []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(images)
        logits = logits.float()
        total_loss += criterion(logits, targets).item()
        labels.append(targets.cpu().numpy())
        probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    y_true = np.concatenate(labels)
    return total_loss / max(len(y_true), 1), y_true, np.concatenate(probs)


def format_summary(metrics: dict, mode: str) -> str:
    """Human-readable one-block summary of the most important metrics."""
    lines = [f"ROC-AUC:            {metrics['auc']:.4f}"]
    if mode == "binary":
        keys = ["sensitivity", "specificity", "precision", "f1"]
    else:
        keys = ["melanoma_recall", "melanoma_specificity", "melanoma_auc", "macro_f1"]
    keys += ["balanced_accuracy", "accuracy"]
    for key in keys:
        if key in metrics:
            lines.append(f"{key + ':':<20}{metrics[key]:.4f}")
    if "recall_per_class" in metrics:
        lines.append("Recall per class:")
        for name, value in metrics["recall_per_class"].items():
            lines.append(f"  {name:<8}{value:.4f}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--device", default=None, help="cuda or cpu (default: auto)")
    parser.add_argument(
        "--target-sensitivity",
        type=float,
        default=None,
        help="e.g. 0.9: additionally report a 'flag as malignant/melanoma' rule whose "
        "probability threshold is chosen on the validation split to reach this sensitivity",
    )
    args = parser.parse_args(argv)

    device = get_device(args.device)
    model, ckpt = load_checkpoint(args.checkpoint, device)
    cfg, class_names = ckpt["config"], ckpt["class_names"]
    mode = cfg["task"]["mode"]

    # Same config + same seed -> exactly the split that was used for training.
    splits, _ = prepare_data(cfg)
    loaders = build_dataloaders(cfg, splits)
    amp = cfg["train"]["amp"] and device.type == "cuda"
    loss, y_true, y_prob = predict_loader(model, loaders[args.split], device, amp)

    metrics = compute_metrics(y_true, y_prob, class_names, mode, cfg["task"]["threshold"])
    metrics["loss"] = loss
    metrics["num_images"] = int(len(y_true))
    metrics["split"] = args.split

    # Optional: one-vs-rest decision rule tuned for a target sensitivity.
    # The threshold is always chosen on the validation split, so the numbers
    # reported for the test split stay unbiased.
    if args.target_sensitivity is not None:
        pos_idx, pos_name = positive_class(class_names, mode)
        if args.split == "val":
            val_true, val_prob = y_true, y_prob
        else:
            _, val_true, val_prob = predict_loader(model, loaders["val"], device, amp)
        threshold = threshold_for_sensitivity(
            val_true == pos_idx, val_prob[:, pos_idx], args.target_sensitivity
        )
        metrics["flag_rule"] = {
            "positive_class": pos_name,
            "target_sensitivity": args.target_sensitivity,
            "threshold": threshold,
            "threshold_chosen_on": "val",
            "val": flag_metrics(val_true == pos_idx, val_prob[:, pos_idx], threshold),
            args.split: flag_metrics(y_true == pos_idx, y_prob[:, pos_idx], threshold),
        }

    if mode == "binary":
        y_pred = (y_prob[:, 1] >= cfg["task"]["threshold"]).astype(int)
    else:
        y_pred = y_prob.argmax(axis=1)
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    out_dir = Path(args.checkpoint).parent / f"eval_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    (out_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    save_confusion_matrix(
        np.array(metrics["confusion_matrix"]), class_names, out_dir / "confusion_matrix.png"
    )

    print(f"\n{args.split} split: {len(y_true)} images, checkpoint epoch {ckpt['epoch']}")
    print(format_summary(metrics, mode))
    print("\n" + report)
    if "flag_rule" in metrics:
        rule = metrics["flag_rule"]
        result = rule[args.split]
        print(
            f"Flag rule: p({rule['positive_class']}) >= {rule['threshold']:.4f} "
            f"(chosen on val for sensitivity >= {rule['target_sensitivity']})\n"
            f"  val : sensitivity {rule['val']['sensitivity']:.4f} | "
            f"specificity {rule['val']['specificity']:.4f}\n"
            f"  {args.split:<4}: sensitivity {result['sensitivity']:.4f} | "
            f"specificity {result['specificity']:.4f} | precision {result['precision']:.4f} | "
            f"missed {result['false_negatives']} | false alarms {result['false_positives']}\n"
        )
    print(f"Results written to {out_dir}")
    return metrics


if __name__ == "__main__":
    main()

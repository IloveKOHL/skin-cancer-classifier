"""Shared helpers: config handling, seeding, device selection and metrics."""

from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from sklearn.metrics import confusion_matrix, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


# --------------------------------------------------------------------------- config


def _deep_merge(base: dict, update: dict) -> dict:
    """Recursively merge ``update`` into a copy of ``base``."""
    merged = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> dict:
    """Load ``configs/default.yaml``, merge ``path`` on top, then apply CLI overrides.

    Overrides use dotted keys, e.g. ``["train.epochs=5", "task.mode=multiclass"]``.
    """
    with open(DEFAULT_CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if path is not None and Path(path).resolve() != DEFAULT_CONFIG:
        with open(path, encoding="utf-8") as f:
            cfg = _deep_merge(cfg, yaml.safe_load(f) or {})

    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key.subkey=value, got: {item!r}")
        dotted, raw = item.split("=", 1)
        *parents, leaf = dotted.split(".")
        node = cfg
        for key in parents:
            if not isinstance(node.get(key), dict):
                raise KeyError(f"Unknown config section in override: {dotted!r}")
            node = node[key]
        if leaf not in node:
            raise KeyError(f"Unknown config key in override: {dotted!r}")
        node[leaf] = yaml.safe_load(raw)

    validate_config(cfg)
    return cfg


def validate_config(cfg: dict) -> None:
    """Fail early on typos in the options that select behaviour."""
    choices = {
        ("task", "mode"): {"binary", "multiclass"},
        ("train", "imbalance"): {"none", "weighted_loss", "weighted_sampler"},
        ("train", "monitor"): {"auc", "recall"},
    }
    for (section, key), allowed in choices.items():
        if cfg[section][key] not in allowed:
            raise ValueError(
                f"{section}.{key} must be one of {sorted(allowed)}, got {cfg[section][key]!r}"
            )


def save_config(cfg: dict, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


# --------------------------------------------------------------------------- runtime


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (CPU and GPU)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic cuDNN kernels: slightly slower, but runs are repeatable.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """Give every DataLoader worker its own reproducible NumPy/random seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_device(name: str | None = None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- metrics


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else float("nan")


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray, **kwargs: Any) -> float:
    """ROC-AUC that returns NaN instead of raising when a class is missing."""
    try:
        return float(roc_auc_score(y_true, y_score, **kwargs))
    except ValueError:
        return float("nan")


def _one_vs_rest(cm: np.ndarray, idx: int) -> dict[str, float]:
    """Sensitivity, specificity, precision and F1 for one class of a confusion matrix."""
    tp = cm[idx, idx]
    fn = cm[idx].sum() - tp
    fp = cm[:, idx].sum() - tp
    tn = cm.sum() - tp - fn - fp
    recall = _safe_div(tp, tp + fn)
    precision = _safe_div(tp, tp + fp)
    f1 = _safe_div(2 * tp, 2 * tp + fp + fn)
    return {
        "sensitivity": recall,
        "specificity": _safe_div(tn, tn + fp),
        "precision": precision,
        "f1": f1,
    }


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
    mode: str,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute classification metrics from class probabilities.

    Args:
        y_true: integer labels, shape (N,).
        y_prob: class probabilities, shape (N, num_classes).
        class_names: label names in index order.
        mode: "binary" (index 1 = malignant) or "multiclass".
        threshold: malignant decision threshold, binary mode only.

    The returned dict always contains ``auc`` and ``recall`` (the two metrics the
    training loop can monitor) plus ``confusion_matrix`` as a nested list.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    labels = list(range(len(class_names)))

    if mode == "binary":
        y_pred = (y_prob[:, 1] >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        metrics: dict[str, Any] = _one_vs_rest(cm, 1)
        metrics["auc"] = _safe_auc(y_true, y_prob[:, 1])
        metrics["recall"] = metrics["sensitivity"]
    else:
        y_pred = y_prob.argmax(axis=1)
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        per_class = {name: _one_vs_rest(cm, i) for i, name in enumerate(class_names)}
        metrics = {
            "auc": _safe_auc(y_true, y_prob, multi_class="ovr", average="macro", labels=labels),
            "macro_f1": float(np.nanmean([m["f1"] for m in per_class.values()])),
            "recall_per_class": {name: m["sensitivity"] for name, m in per_class.items()},
            "per_class": per_class,
        }
        if "mel" in class_names:
            mel = class_names.index("mel")
            metrics["melanoma_recall"] = per_class["mel"]["sensitivity"]
            metrics["melanoma_specificity"] = per_class["mel"]["specificity"]
            metrics["melanoma_auc"] = _safe_auc((y_true == mel).astype(int), y_prob[:, mel])
            metrics["recall"] = metrics["melanoma_recall"]
        else:
            metrics["recall"] = float("nan")

    recalls = [_safe_div(cm[i, i], cm[i].sum()) for i in labels]
    metrics["accuracy"] = _safe_div(np.trace(cm), cm.sum())
    metrics["balanced_accuracy"] = float(np.nanmean(recalls))
    metrics["confusion_matrix"] = cm.tolist()
    return metrics


def save_confusion_matrix(cm: np.ndarray, class_names: list[str], path: str | Path) -> None:
    """Save a confusion matrix (counts + row-normalised colours) as PNG."""
    import matplotlib

    matplotlib.use("Agg")  # no display needed
    import matplotlib.pyplot as plt

    cm = np.asarray(cm)
    row_sums = cm.sum(axis=1, keepdims=True)
    normalised = np.divide(cm, row_sums, out=np.zeros(cm.shape, dtype=float), where=row_sums > 0)

    size = max(4.5, 0.9 * len(class_names) + 2)
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.imshow(normalised, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (colour = recall per true class)")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            colour = "white" if normalised[i, j] > 0.5 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=colour)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

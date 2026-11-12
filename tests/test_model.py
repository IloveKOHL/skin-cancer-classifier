from __future__ import annotations

import numpy as np
import pytest
import torch

from src.gradcam import grad_cam
from src.model import build_model
from src.utils import compute_metrics, flag_metrics, load_config, threshold_for_sensitivity


@pytest.mark.parametrize("num_classes", [2, 7])
def test_forward_output_shape(cfg, num_classes):
    model = build_model(cfg, num_classes=num_classes).eval()
    with torch.no_grad():
        logits = model(torch.randn(2, 3, 64, 64))
    assert logits.shape == (2, num_classes)
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize("name", ["resnet18", "convnext_atto"])
def test_backbone_is_swappable(cfg, name):
    """Small relatives of ResNet50 / ConvNeXt-Tiny keep the test fast."""
    cfg["model"]["name"] = name
    model = build_model(cfg, num_classes=7).eval()
    with torch.no_grad():
        assert model(torch.randn(1, 3, 64, 64)).shape == (1, 7)


def test_grad_cam_shape_and_range(cfg):
    model = build_model(cfg, num_classes=2)
    cam, probs = grad_cam(model, torch.randn(1, 3, 64, 64))
    assert cam.shape == (64, 64)
    assert cam.min() >= 0.0 and cam.max() <= 1.0
    assert probs.sum() == pytest.approx(1.0, abs=1e-5)


def test_binary_metrics_on_known_example():
    # 4 malignant (3 found), 6 benign (4 correctly rejected)
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    p_mal = np.array([0.9, 0.8, 0.7, 0.2, 0.6, 0.6, 0.1, 0.1, 0.1, 0.1])
    metrics = compute_metrics(
        y_true, np.stack([1 - p_mal, p_mal], axis=1), ["benign", "malignant"], "binary"
    )
    assert metrics["sensitivity"] == pytest.approx(3 / 4)
    assert metrics["specificity"] == pytest.approx(4 / 6)
    assert metrics["precision"] == pytest.approx(3 / 5)
    assert metrics["accuracy"] == pytest.approx(7 / 10)
    assert metrics["confusion_matrix"] == [[4, 2], [1, 3]]
    assert 0.5 < metrics["auc"] <= 1.0


def test_multiclass_metrics_report_melanoma_recall():
    class_names = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
    y_true = np.array([4, 4, 5, 5, 0, 1, 2, 3, 6])
    y_pred = np.array([4, 5, 5, 5, 0, 1, 2, 3, 6])  # one melanoma missed as nevus
    metrics = compute_metrics(y_true, np.eye(7)[y_pred], class_names, "multiclass")
    assert metrics["melanoma_recall"] == pytest.approx(0.5)
    assert metrics["recall"] == metrics["melanoma_recall"]
    assert metrics["recall_per_class"]["nv"] == pytest.approx(1.0)
    assert set(metrics["recall_per_class"]) == set(class_names)


def test_threshold_for_target_sensitivity():
    y_true = np.array([1] * 10 + [0] * 10)
    scores = np.concatenate([np.linspace(0.1, 1.0, 10), np.linspace(0.0, 0.45, 10)])
    threshold = threshold_for_sensitivity(y_true, scores, target=0.9)
    assert threshold == pytest.approx(0.2)  # 9 of 10 positives have a score >= 0.2
    result = flag_metrics(y_true, scores, threshold)
    assert result["sensitivity"] == pytest.approx(0.9)
    assert result["false_negatives"] == 1
    # a stricter target can only lower the threshold
    assert threshold_for_sensitivity(y_true, scores, target=1.0) <= threshold


def test_config_overrides_and_validation():
    cfg = load_config(overrides=["task.mode=multiclass", "train.batch_size=16"])
    assert cfg["task"]["mode"] == "multiclass"
    assert cfg["train"]["batch_size"] == 16
    with pytest.raises(ValueError):
        load_config(overrides=["task.mode=ternary"])
    with pytest.raises(KeyError):
        load_config(overrides=["train.not_a_key=1"])

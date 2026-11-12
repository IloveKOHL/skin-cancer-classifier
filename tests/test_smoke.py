"""End-to-end smoke test on synthetic images: train -> evaluate -> Grad-CAM."""

from __future__ import annotations

import json

import pytest

from src import evaluate, gradcam, train
from src.utils import save_config


@pytest.mark.parametrize("mode", ["binary", "multiclass"])
def test_train_evaluate_gradcam(cfg, tmp_path, mode):
    cfg["task"]["mode"] = mode
    config_path = tmp_path / "config.yaml"
    save_config(cfg, config_path)

    run_dir = train.main(["--config", str(config_path), "--device", "cpu"])
    checkpoint = run_dir / "best.pt"
    assert checkpoint.exists()
    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "splits.csv").exists()

    metrics = evaluate.main(
        ["--checkpoint", str(checkpoint), "--device", "cpu", "--target-sensitivity", "0.9"]
    )
    assert metrics["flag_rule"]["val"]["sensitivity"] >= 0.9
    assert 0.0 <= metrics["flag_rule"]["test"]["sensitivity"] <= 1.0
    eval_dir = run_dir / "eval_test"
    assert (eval_dir / "confusion_matrix.png").stat().st_size > 0
    assert (eval_dir / "classification_report.txt").exists()
    saved = json.loads((eval_dir / "metrics.json").read_text())
    assert saved["num_images"] == metrics["num_images"] > 0
    assert ("sensitivity" in saved) if mode == "binary" else ("melanoma_recall" in saved)

    out_dir = gradcam.main(
        ["--checkpoint", str(checkpoint), "--num-samples", "2", "--device", "cpu"]
    )
    assert len(list(out_dir.glob("*.png"))) == 2

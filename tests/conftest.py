"""Shared fixtures: a tiny synthetic dataset and a config that points to it."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.make_dummy_data import make_dummy_dataset  # noqa: E402
from src.utils import load_config  # noqa: E402


@pytest.fixture(scope="session")
def dummy_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("dummy_data")
    make_dummy_dataset(root, lesions_per_class=10, image_size=64)
    return root


@pytest.fixture
def cfg(dummy_root, tmp_path) -> dict:
    """Small CPU-friendly config (no pretrained weights -> no network access)."""
    config = load_config()
    config["data"].update(
        root=str(dummy_root),
        metadata_csv=str(dummy_root / "HAM10000_metadata.csv"),
        image_size=64,
        num_workers=0,
    )
    config["model"]["pretrained"] = False
    config["train"].update(epochs=1, batch_size=8, warmup_epochs=0)
    config["logging"].update(runs_dir=str(tmp_path / "runs"), run_name="test")
    return config

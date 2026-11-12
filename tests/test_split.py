"""The split must be leakage-free: no lesion may appear in more than one split."""

from __future__ import annotations

import pandas as pd

from src.dataset import HAM_CLASSES, load_metadata, split_by_lesion


def _splits(cfg, seed=42):
    df = load_metadata(cfg["data"]["metadata_csv"], cfg["data"]["root"], "multiclass")
    return df, split_by_lesion(df, val_size=0.15, test_size=0.15, seed=seed)


def test_dummy_data_has_multi_image_lesions(cfg):
    """Sanity check of the fixture: without repeated lesions the leakage test is vacuous."""
    df, _ = _splits(cfg)
    assert df["lesion_id"].duplicated().any()


def test_no_lesion_in_multiple_splits(cfg):
    _, splits = _splits(cfg)
    lesions = {name: set(part["lesion_id"]) for name, part in splits.items()}
    assert lesions["train"].isdisjoint(lesions["val"])
    assert lesions["train"].isdisjoint(lesions["test"])
    assert lesions["val"].isdisjoint(lesions["test"])


def test_every_image_is_used_exactly_once(cfg):
    df, splits = _splits(cfg)
    all_ids = pd.concat([part["image_id"] for part in splits.values()])
    assert not all_ids.duplicated().any()
    assert set(all_ids) == set(df["image_id"])


def test_split_is_deterministic_for_a_seed(cfg):
    _, first = _splits(cfg, seed=42)
    _, second = _splits(cfg, seed=42)
    _, other = _splits(cfg, seed=7)
    for name in ("train", "val", "test"):
        assert first[name]["image_id"].tolist() == second[name]["image_id"].tolist()
    assert first["test"]["image_id"].tolist() != other["test"]["image_id"].tolist()


def test_split_is_stratified(cfg):
    """Every class must be present in every split."""
    _, splits = _splits(cfg)
    for name, part in splits.items():
        assert set(part["dx"]) == set(HAM_CLASSES), f"class missing in {name}"


def test_split_sizes_are_roughly_as_requested(cfg):
    df, splits = _splits(cfg)
    n_lesions = df["lesion_id"].nunique()
    for name, expected in (("val", 0.15), ("test", 0.15)):
        fraction = splits[name]["lesion_id"].nunique() / n_lesions
        assert abs(fraction - expected) < 0.05

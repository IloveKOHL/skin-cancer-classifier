from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from src.dataset import (
    HAM_CLASSES,
    MALIGNANT,
    SkinLesionDataset,
    build_dataloaders,
    build_transforms,
    compute_class_weights,
    load_metadata,
    prepare_data,
)


def test_binary_labels_follow_malignant_grouping(cfg):
    df = load_metadata(cfg["data"]["metadata_csv"], cfg["data"]["root"], "binary")
    assert set(df["label"]) == {0, 1}
    assert (df["label"] == df["dx"].isin(MALIGNANT).astype(int)).all()


def test_multiclass_labels_match_class_order(cfg):
    df = load_metadata(cfg["data"]["metadata_csv"], cfg["data"]["root"], "multiclass")
    assert (df["label"].map(lambda i: HAM_CLASSES[i]) == df["dx"]).all()


def test_missing_images_raise_a_clear_error(cfg, tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        load_metadata(cfg["data"]["metadata_csv"], tmp_path, "binary")


@pytest.mark.parametrize("train", [True, False])
def test_dataset_item_shape_and_type(cfg, train):
    df = load_metadata(cfg["data"]["metadata_csv"], cfg["data"]["root"], "binary")
    image, label = SkinLesionDataset(df, build_transforms(cfg, train=train))[0]
    assert image.shape == (3, 64, 64)
    assert image.dtype == torch.float32
    assert label in (0, 1)


def test_eval_transform_is_deterministic(cfg):
    df = load_metadata(cfg["data"]["metadata_csv"], cfg["data"]["root"], "binary")
    dataset = SkinLesionDataset(df, build_transforms(cfg, train=False))
    assert torch.equal(dataset[0][0], dataset[0][0])


def test_class_weights_favour_rare_classes():
    weights = compute_class_weights(np.array([0] * 90 + [1] * 10), num_classes=2)
    assert weights[1] > weights[0]
    assert weights.tolist() == pytest.approx([100 / 180, 100 / 20])


def _write_extra(cfg, tmp_path, rows):
    """Create an extra-training CSV that reuses image files of the dummy dataset."""
    path = tmp_path / "extra.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    cfg["data"]["extra_train_csv"] = str(path)
    cfg["data"]["extra_train_root"] = str(tmp_path / "extra_images")


def test_extra_data_goes_to_train_only(cfg, tmp_path):
    base, _ = prepare_data(cfg)
    source = Image.open(base["train"]["path"].iloc[0])
    (tmp_path / "extra_images").mkdir()
    for name in ("EXTRA_1", "EXTRA_2"):
        source.save(tmp_path / "extra_images" / f"{name}.jpg")
    _write_extra(cfg, tmp_path, [
        {"lesion_id": "X1", "image_id": "EXTRA_1", "dx": "mel"},
        {"lesion_id": "X2", "image_id": "EXTRA_2", "dx": "nv"},
    ])
    splits, _ = prepare_data(cfg)
    assert len(splits["train"]) == len(base["train"]) + 2
    assert {"EXTRA_1", "EXTRA_2"} <= set(splits["train"]["image_id"])
    for name in ("val", "test"):  # evaluation splits are untouched
        assert splits[name]["image_id"].tolist() == base[name]["image_id"].tolist()


def test_extra_data_overlapping_main_dataset_is_rejected(cfg, tmp_path):
    base, _ = prepare_data(cfg)
    leaked = base["test"].iloc[0]
    (tmp_path / "extra_images").mkdir()
    Image.open(leaked["path"]).save(tmp_path / "extra_images" / f"{leaked['image_id']}.jpg")
    _write_extra(cfg, tmp_path, [
        {"lesion_id": "X1", "image_id": leaked["image_id"], "dx": leaked["dx"]},
    ])
    with pytest.raises(ValueError, match="leak"):
        prepare_data(cfg)


@pytest.mark.parametrize("imbalance", ["none", "weighted_loss", "weighted_sampler"])
def test_dataloaders_yield_batches(cfg, imbalance):
    cfg["train"]["imbalance"] = imbalance
    splits, class_names = prepare_data(cfg)
    loaders = build_dataloaders(cfg, splits)
    images, labels = next(iter(loaders["train"]))
    assert images.shape == (8, 3, 64, 64)
    assert labels.max() < len(class_names)
    assert sum(len(batch[1]) for batch in loaders["val"]) == len(splits["val"])

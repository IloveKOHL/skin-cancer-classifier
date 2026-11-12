"""The ISIC 2019 preparation must drop every HAM10000 image (leakage guard)."""

from __future__ import annotations

import pandas as pd

from scripts.prepare_isic2019 import build_extra_table


def test_ham_images_and_unmappable_classes_are_dropped():
    cols = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"]

    def row(image, label):
        return {"image": image, **{c: float(c == label) for c in cols}}

    labels = pd.DataFrame([
        row("ISIC_0024306", "NV"),            # part of HAM10000 -> must be dropped
        row("ISIC_0000001", "MEL"),
        row("ISIC_0000002_downsampled", "AK"),
        row("ISIC_0000003", "SCC"),           # no HAM10000 class -> dropped
        row("ISIC_0000004", "BKL"),
    ])
    metadata = pd.DataFrame({
        "image": labels["image"],
        "age_approx": [50, 60, 70, 80, 40],
        "anatom_site_general": ["head/neck"] * 5,
        "lesion_id": ["HAM_1", "BCN_1", None, "BCN_2", "BCN_1"],
        "sex": ["male"] * 5,
    })
    extra, stats = build_extra_table(labels, metadata, ham_ids={"ISIC_0024306"})

    assert extra["image_id"].tolist() == [
        "ISIC_0000001", "ISIC_0000002_downsampled", "ISIC_0000004",
    ]
    assert extra["dx"].tolist() == ["mel", "akiec", "bkl"]
    assert extra["lesion_id"].tolist() == ["BCN_1", "ISIC2019_ISIC_0000002_downsampled", "BCN_1"]
    assert stats == {
        "total": 5, "dropped_ham10000": 1, "dropped_scc_or_unknown": 1, "kept": 3,
    }

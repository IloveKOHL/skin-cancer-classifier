"""Create a tiny synthetic dataset in HAM10000 format for smoke tests.

Usage:
    python scripts/make_dummy_data.py --output data/dummy

The images are random coloured blobs. They only exist to check that the code
runs end to end - any metric computed on them is meaningless.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

HAM_CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


def make_dummy_dataset(
    output: str | Path, lesions_per_class: int = 12, image_size: int = 96, seed: int = 0
) -> Path:
    """Write images + HAM10000_metadata.csv to ``output`` and return the CSV path."""
    rng = np.random.default_rng(seed)
    output = Path(output)
    image_dir = output / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    lesion_no = image_no = 0
    for class_idx, dx in enumerate(HAM_CLASSES):
        for _ in range(lesions_per_class):
            lesion_id = f"HAM_{lesion_no:07d}"
            lesion_no += 1
            # Like the real dataset: some lesions were photographed more than once.
            for _ in range(int(rng.integers(1, 4))):
                image_id = f"ISIC_{image_no:07d}"
                image_no += 1
                skin = rng.integers(170, 230, size=3)
                noise = rng.integers(-12, 12, size=(image_size, image_size, 3))
                array = np.clip(skin + noise, 0, 255).astype(np.uint8)
                image = Image.fromarray(array)
                # A blob whose colour depends weakly on the class.
                colour = tuple(int(c) for c in (60 + 25 * class_idx, 40, 30 + 20 * class_idx))
                cx, cy = rng.integers(image_size // 3, 2 * image_size // 3, size=2)
                r = int(rng.integers(image_size // 8, image_size // 4))
                ImageDraw.Draw(image).ellipse([cx - r, cy - r, cx + r, cy + r], fill=colour)
                image.save(image_dir / f"{image_id}.jpg", quality=90)
                rows.append(
                    {
                        "lesion_id": lesion_id,
                        "image_id": image_id,
                        "dx": dx,
                        "dx_type": "histo",
                        "age": float(rng.integers(20, 85)),
                        "sex": str(rng.choice(["male", "female"])),
                        "localization": str(rng.choice(["back", "face", "trunk"])),
                    }
                )

    csv_path = output / "HAM10000_metadata.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", default="data/dummy")
    parser.add_argument("--lesions-per-class", type=int, default=12)
    args = parser.parse_args()
    path = make_dummy_dataset(args.output, args.lesions_per_class)
    print(f"Dummy dataset written: {path}")

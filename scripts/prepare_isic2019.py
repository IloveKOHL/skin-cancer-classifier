"""Download ISIC 2019 and prepare it as EXTRA TRAINING data (about 9.8 GB download).

Usage:
    python scripts/prepare_isic2019.py            # -> data/ISIC2019
    python -m src.train --override data.extra_train_csv=data/ISIC2019/extra_train.csv \
        data.extra_train_root=data/ISIC2019/images

What it does:
  1. Downloads the ISIC 2019 training images, labels and metadata.
  2. Drops every image that is part of HAM10000. ISIC 2019 contains HAM10000, and
     the HAM10000 validation/test images must never end up in the training set
     (data leakage). The remaining images come from other hospitals (BCN_20000,
     MSK) and are only ever used for training.
  3. Maps the labels to the 7 HAM10000 classes. Squamous cell carcinoma (SCC) and
     "unknown" have no HAM10000 class and are dropped.
  4. Stores downscaled copies (shorter side 450 px, like HAM10000) so training
     does not have to decode large originals.

Source:  https://challenge.isic-archive.com/data/#2019
License: CC BY-NC 4.0 (non-commercial use, attribution required). Please cite:
  - BCN_20000: Hernandez-Perez et al., Sci. Data (2024).
  - HAM10000:  Tschandl et al., Sci. Data 5, 180161 (2018).
  - MSK:       Codella et al., arXiv:1710.05006 (2017).
"""

from __future__ import annotations

import argparse
import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

BASE_URL = "https://isic-archive.s3.amazonaws.com/challenges/2019/"
ZIP_NAME = "ISIC_2019_Training_Input.zip"
LABELS_NAME = "ISIC_2019_Training_GroundTruth.csv"
METADATA_NAME = "ISIC_2019_Training_Metadata.csv"
HEADERS = {"User-Agent": "skin-cancer-classifier-downloader"}

# ISIC 2019 label column -> HAM10000 class code
LABEL_MAP = {
    "MEL": "mel", "NV": "nv", "BCC": "bcc", "AK": "akiec",
    "BKL": "bkl", "DF": "df", "VASC": "vasc",
}  # fmt: skip


def download(url: str, target: Path) -> None:
    """Download with a progress bar; an interrupted download is resumed."""
    if target.exists():
        print(f"Already present: {target}")
        return
    partial = target.with_suffix(target.suffix + ".part")
    start = partial.stat().st_size if partial.exists() else 0
    headers = dict(HEADERS, Range=f"bytes={start}-") if start else HEADERS
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        if start and response.status != 206:  # server ignored the range -> start over
            start = 0
        remaining = int(response.headers.get("Content-Length") or 0)
        with open(partial, "ab" if start else "wb") as out, tqdm(
            total=start + remaining or None, initial=start, unit="B", unit_scale=True,
            desc=target.name, mininterval=5,
        ) as bar:
            while chunk := response.read(1 << 20):
                out.write(chunk)
                bar.update(len(chunk))
    partial.replace(target)


def build_extra_table(labels: pd.DataFrame, metadata: pd.DataFrame, ham_ids: set[str]):
    """Return the HAM-format table of usable non-HAM10000 images."""
    label_cols = [c for c in labels.columns if c != "image"]
    table = labels[["image"]].copy()
    table["isic_label"] = labels[label_cols].idxmax(axis=1)
    table = table.merge(metadata, on="image", how="left")

    in_ham = table["image"].isin(ham_ids)
    mappable = table["isic_label"].isin(LABEL_MAP)
    extra = table[~in_ham & mappable].copy()
    stats = {
        "total": len(table),
        "dropped_ham10000": int(in_ham.sum()),
        "dropped_scc_or_unknown": int((~in_ham & ~mappable).sum()),
        "kept": len(extra),
    }
    out = pd.DataFrame(
        {
            # images without a lesion id are treated as their own lesion
            "lesion_id": extra["lesion_id"].fillna("ISIC2019_" + extra["image"]),
            "image_id": extra["image"],
            "dx": extra["isic_label"].map(LABEL_MAP),
            "age": extra["age_approx"],
            "sex": extra["sex"],
            "localization": extra["anatom_site_general"],
            "source": "ISIC2019",
        }
    )
    return out.reset_index(drop=True), stats


def extract_resized(zip_path: Path, image_ids: set[str], image_dir: Path, short_side: int) -> int:
    """Write downscaled copies of the selected images. Returns the number written."""
    image_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    with zipfile.ZipFile(zip_path) as archive:
        members = [m for m in archive.namelist() if Path(m).stem in image_ids]
        for member in tqdm(members, desc="resize", mininterval=5):
            target = image_dir / f"{Path(member).stem}.jpg"
            if target.exists():
                continue
            image = Image.open(io.BytesIO(archive.read(member))).convert("RGB")
            scale = short_side / min(image.size)
            if scale < 1:
                size = (round(image.width * scale), round(image.height * scale))
                image = image.resize(size, Image.LANCZOS)
            image.save(target, quality=92)
            written += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", default="data/ISIC2019")
    parser.add_argument("--ham-metadata", default="data/HAM10000/HAM10000_metadata.csv")
    parser.add_argument("--short-side", type=int, default=450)
    parser.add_argument("--keep-zip", action="store_true", help="Keep the 9.8 GB zip afterwards")
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for name in (LABELS_NAME, METADATA_NAME, ZIP_NAME):
        download(BASE_URL + name, out / name)

    ham = pd.read_csv(args.ham_metadata, sep=None, engine="python")
    extra, stats = build_extra_table(
        pd.read_csv(out / LABELS_NAME), pd.read_csv(out / METADATA_NAME), set(ham["image_id"])
    )
    print("ISIC 2019 images:", stats)
    print(extra["dx"].value_counts().to_string())

    extract_resized(out / ZIP_NAME, set(extra["image_id"]), out / "images", args.short_side)
    found = {p.stem for p in (out / "images").glob("*.jpg")}
    extra = extra[extra["image_id"].isin(found)]
    extra.to_csv(out / "extra_train.csv", index=False)
    print(f"\nWrote {len(extra)} rows to {out / 'extra_train.csv'}")
    if not args.keep_zip:
        (out / ZIP_NAME).unlink()


if __name__ == "__main__":
    main()

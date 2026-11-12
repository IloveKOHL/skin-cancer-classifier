"""Download HAM10000 from the Harvard Dataverse (about 2.8 GB).

Usage:
    python scripts/download_data.py                 # everything -> data/HAM10000
    python scripts/download_data.py --skip-images   # metadata only (< 1 MB)

Source:  https://doi.org/10.7910/DVN/DBW86T
License: CC BY-NC 4.0 (non-commercial use, attribution required). By running
this script you agree to the dataset's terms of use. Please cite:
    Tschandl, P., Rosendahl, C. & Kittler, H. The HAM10000 dataset, a large
    collection of multi-source dermatoscopic images of common pigmented skin
    lesions. Sci. Data 5, 180161 (2018).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

from tqdm import tqdm

API = "https://dataverse.harvard.edu/api"
DOI = "doi:10.7910/DVN/DBW86T"
METADATA_FILE = "HAM10000_metadata.tab"
IMAGE_ZIPS = ["HAM10000_images_part_1.zip", "HAM10000_images_part_2.zip"]
HEADERS = {"User-Agent": "skin-cancer-classifier-downloader"}


def _open(url: str):
    return urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=60)


def list_files() -> dict[str, dict]:
    """Map file name -> Dataverse file record (id, size, md5) for the dataset."""
    with _open(f"{API}/datasets/:persistentId/?persistentId={DOI}") as response:
        dataset = json.load(response)
    files = dataset["data"]["latestVersion"]["files"]
    return {f["dataFile"]["filename"]: f["dataFile"] for f in files}


def download(record: dict, target: Path, original: bool = False) -> None:
    """Stream one file to ``target`` with a progress bar and verify its MD5."""
    url = f"{API}/access/datafile/{record['id']}"
    if original:
        # Dataverse converts tabular files; ask for the file as it was uploaded.
        url += "?format=original"
    md5 = hashlib.md5()
    partial = target.with_suffix(target.suffix + ".part")
    with _open(url) as response, open(partial, "wb") as out:
        total = int(response.headers.get("Content-Length") or 0) or None
        with tqdm(total=total, unit="B", unit_scale=True, desc=target.name) as bar:
            while chunk := response.read(1 << 20):
                out.write(chunk)
                md5.update(chunk)
                bar.update(len(chunk))
    expected = record.get("md5")
    if expected and not original and md5.hexdigest() != expected:
        partial.unlink()
        raise RuntimeError(f"MD5 mismatch for {target.name} - please run the script again.")
    partial.replace(target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", default="data/HAM10000")
    parser.add_argument("--skip-images", action="store_true", help="Download the metadata only")
    parser.add_argument("--keep-zips", action="store_true", help="Do not delete zips after unpacking")
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    files = list_files()

    metadata = out / "HAM10000_metadata.csv"
    if metadata.exists():
        print(f"Already present: {metadata}")
    else:
        download(files[METADATA_FILE], metadata, original=True)

    if not args.skip_images:
        for name in IMAGE_ZIPS:
            folder = out / Path(name).stem
            if folder.exists() and any(folder.rglob("*.jpg")):
                print(f"Already present: {folder}")
                continue
            zip_path = out / name
            if not zip_path.exists():
                download(files[name], zip_path)
            print(f"Unpacking {name} ...")
            with zipfile.ZipFile(zip_path) as archive:
                archive.extractall(folder)
            if not args.keep_zips:
                zip_path.unlink()

    n_images = sum(1 for _ in out.rglob("*.jpg"))
    print(f"\nDone. {n_images} images below {out} (expected 10015 for the full dataset).")


if __name__ == "__main__":
    main()

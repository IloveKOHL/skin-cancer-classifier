# Data

This folder is intentionally empty in Git. **No images and no model weights are
committed to this repository** (everything in `data/` except this file is ignored).

## HAM10000

| | |
|---|---|
| Source | Harvard Dataverse, <https://doi.org/10.7910/DVN/DBW86T> (also part of the [ISIC Archive](https://www.isic-archive.com/)) |
| Content | 10,015 dermatoscopic images of 7,470 lesions, 7 diagnostic classes |
| License | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) - non-commercial use only, attribution required |
| Citation | Tschandl, P., Rosendahl, C. & Kittler, H. *The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions.* Sci. Data 5, 180161 (2018). <https://doi.org/10.1038/sdata.2018.161> |

Read the terms of use on the Dataverse page before downloading. The license applies
to the images and to anything derived from them that you redistribute.

### Option A - download script (recommended)

```bash
python scripts/download_data.py
```

Downloads about 2.8 GB from the Harvard Dataverse, verifies the checksums and
unpacks to `data/HAM10000/`. Run it again if it is interrupted; finished parts
are skipped.

### Option B - manual download from the Dataverse

1. Open <https://doi.org/10.7910/DVN/DBW86T>.
2. Download `HAM10000_images_part_1.zip`, `HAM10000_images_part_2.zip` and
   `HAM10000_metadata` (choose "Original File Format" / CSV).
3. Unpack both zips into `data/HAM10000/` and save the metadata as
   `data/HAM10000/HAM10000_metadata.csv`.

### Option C - Kaggle mirror

```bash
pip install kaggle   # needs an API token in ~/.kaggle/kaggle.json
kaggle datasets download -d kmader/skin-cancer-mnist-ham10000 -p data/HAM10000 --unzip
```

The Kaggle mirror is a third-party upload of the same data; the CC BY-NC 4.0
license still applies.

### Expected layout

```
data/HAM10000/
  HAM10000_metadata.csv
  HAM10000_images_part_1/   ISIC_0024306.jpg ...
  HAM10000_images_part_2/   ISIC_0029306.jpg ...
```

Folder names below `data/HAM10000/` do not matter: images are found by searching
recursively for `<image_id>.jpg`. If you use a different location, change
`data.root` and `data.metadata_csv` in `configs/default.yaml`.

### Classes

| Code | Diagnosis | Binary group |
|---|---|---|
| `nv` | Melanocytic nevus | benign |
| `bkl` | Benign keratosis-like lesion | benign |
| `df` | Dermatofibroma | benign |
| `vasc` | Vascular lesion | benign |
| `mel` | Melanoma | malignant |
| `bcc` | Basal cell carcinoma | malignant |
| `akiec` | Actinic keratosis / intraepithelial carcinoma (Bowen's disease) | malignant (pre-cancerous / in situ) |

## Synthetic smoke-test data

```bash
python scripts/make_dummy_data.py --output data/dummy
```

Creates random blob images in the same format. Only for checking that the code runs.

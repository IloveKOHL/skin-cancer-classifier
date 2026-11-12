"""HAM10000 metadata loading, leakage-free splitting, transforms and DataLoaders."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from src.utils import seed_worker

# The 7 diagnostic categories of HAM10000, in a fixed order (= multiclass label index).
HAM_CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

# Binary grouping. akiec (actinic keratosis / Bowen's disease) is pre-cancerous or
# in-situ carcinoma and is commonly counted as malignant in HAM10000 work.
MALIGNANT = {"mel", "bcc", "akiec"}
BINARY_CLASSES = ["benign", "malignant"]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_class_names(mode: str) -> list[str]:
    return BINARY_CLASSES if mode == "binary" else HAM_CLASSES


# --------------------------------------------------------------------------- metadata


def load_metadata(metadata_csv: str | Path, data_root: str | Path, mode: str) -> pd.DataFrame:
    """Read the HAM10000 metadata and add ``path`` and ``label`` columns.

    Images are located by searching ``data_root`` recursively for ``<image_id>.jpg``,
    so both the Dataverse layout (part_1 / part_2 folders) and the Kaggle layout work.
    """
    metadata_csv = Path(metadata_csv)
    if not metadata_csv.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_csv}. See data/README.md for download steps."
        )
    # The Dataverse export is tab-separated (.tab), the Kaggle one comma-separated.
    df = pd.read_csv(metadata_csv, sep=None, engine="python")
    missing = {"lesion_id", "image_id", "dx"} - set(df.columns)
    if missing:
        raise ValueError(f"Metadata is missing required columns: {sorted(missing)}")

    unknown = set(df["dx"]) - set(HAM_CLASSES)
    if unknown:
        raise ValueError(f"Unknown diagnosis codes in metadata: {sorted(unknown)}")

    index = {p.stem: p for p in Path(data_root).rglob("*.jpg")}
    df["path"] = df["image_id"].map(lambda image_id: str(index.get(image_id, "")))
    not_found = int((df["path"] == "").sum())
    if not_found:
        raise FileNotFoundError(
            f"{not_found} of {len(df)} images listed in the metadata were not found "
            f"below {data_root}. See data/README.md for the expected layout."
        )

    if mode == "binary":
        df["label"] = df["dx"].isin(MALIGNANT).astype(int)
    else:
        df["label"] = df["dx"].map(HAM_CLASSES.index)
    return df


# --------------------------------------------------------------------------- split


def split_by_lesion(
    df: pd.DataFrame, val_size: float, test_size: float, seed: int
) -> dict[str, pd.DataFrame]:
    """Split into train/val/test on the lesion level, stratified by diagnosis.

    HAM10000 contains several images of the same lesion. Splitting by image would put
    near-duplicates into train and test (data leakage) and inflate the test metrics.
    Here every ``lesion_id`` is assigned to exactly one split.

    Stratification always uses the 7-class ``dx`` label, so binary and multiclass
    mode share the same split for a given seed.
    """
    lesions = df.groupby("lesion_id", sort=True)["dx"].first().reset_index()

    train_val, test = train_test_split(
        lesions, test_size=test_size, stratify=lesions["dx"], random_state=seed
    )
    train, val = train_test_split(
        train_val,
        test_size=val_size / (1.0 - test_size),  # val_size is relative to the full set
        stratify=train_val["dx"],
        random_state=seed,
    )

    splits = {}
    for name, part in (("train", train), ("val", val), ("test", test)):
        mask = df["lesion_id"].isin(part["lesion_id"])
        splits[name] = df[mask].sort_values("image_id").reset_index(drop=True)
    return splits


# --------------------------------------------------------------------------- transforms


def get_norm_stats(model_name: str) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Mean/std the pretrained timm model expects (falls back to ImageNet values)."""
    try:
        import timm

        cfg = timm.get_pretrained_cfg(model_name)
        return tuple(cfg.mean), tuple(cfg.std)
    except Exception:
        return IMAGENET_MEAN, IMAGENET_STD


def build_transforms(cfg: dict, train: bool) -> transforms.Compose:
    size = cfg["data"]["image_size"]
    mean, std = get_norm_stats(cfg["model"]["name"])
    steps: list = [transforms.Resize((size, size))]

    if train:
        aug = cfg["augmentation"]
        crop_scale = aug.get("random_resized_crop_scale")
        if crop_scale:
            # Random zoom/crop: the lesion is seen at varying size and position.
            steps = [
                transforms.RandomResizedCrop(size, scale=tuple(crop_scale), ratio=(0.9, 1.1))
            ]
        # Dermatoscopic images have no canonical orientation, so flips and
        # rotations are label-preserving.
        if aug["hflip"]:
            steps.append(transforms.RandomHorizontalFlip())
        if aug["vflip"]:
            steps.append(transforms.RandomVerticalFlip())
        if aug["rotation_degrees"]:
            steps.append(transforms.RandomRotation(aug["rotation_degrees"]))
        if aug["color_jitter"]:
            steps.append(transforms.ColorJitter(**aug["color_jitter"]))

    steps += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    return transforms.Compose(steps)


# --------------------------------------------------------------------------- dataset


class SkinLesionDataset(Dataset):
    """Returns ``(image_tensor, label)`` for each row of a metadata DataFrame."""

    def __init__(self, df: pd.DataFrame, transform=None):
        self.paths = df["path"].tolist()
        self.labels = df["label"].astype(int).tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        image = Image.open(self.paths[idx]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[idx]


def compute_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    """Inverse-frequency ("balanced") class weights: N / (K * count_k)."""
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    counts[counts == 0] = np.nan  # a class without samples gets weight 0
    weights = len(labels) / (num_classes * counts)
    return torch.tensor(np.nan_to_num(weights, nan=0.0), dtype=torch.float32)


def build_dataloaders(cfg: dict, splits: dict[str, pd.DataFrame]) -> dict[str, DataLoader]:
    """Create train/val/test loaders. Only the train loader augments and shuffles."""
    num_workers = cfg["data"]["num_workers"]
    common = dict(
        batch_size=cfg["train"]["batch_size"],
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker,
    )
    generator = torch.Generator().manual_seed(cfg["seed"])

    train_ds = SkinLesionDataset(splits["train"], build_transforms(cfg, train=True))
    sampler = None
    if cfg["train"]["imbalance"] == "weighted_sampler":
        # Draw every class with equal probability (with replacement).
        labels = np.array(train_ds.labels)
        class_weights = compute_class_weights(labels, len(get_class_names(cfg["task"]["mode"])))
        sampler = WeightedRandomSampler(
            class_weights[labels].double(), num_samples=len(labels), generator=generator
        )

    loaders = {
        "train": DataLoader(
            train_ds,
            shuffle=sampler is None,
            sampler=sampler,
            drop_last=len(train_ds) > cfg["train"]["batch_size"],
            generator=generator if sampler is None else None,
            **common,
        )
    }
    eval_tf = build_transforms(cfg, train=False)
    for name in ("val", "test"):
        loaders[name] = DataLoader(SkinLesionDataset(splits[name], eval_tf), shuffle=False, **common)
    return loaders


def prepare_data(cfg: dict) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Convenience wrapper: metadata -> lesion-level splits + class names."""
    mode = cfg["task"]["mode"]
    df = load_metadata(cfg["data"]["metadata_csv"], cfg["data"]["root"], mode)
    splits = split_by_lesion(df, cfg["data"]["val_size"], cfg["data"]["test_size"], cfg["seed"])

    extra_csv = cfg["data"].get("extra_train_csv")
    if extra_csv:
        extra = load_metadata(extra_csv, cfg["data"]["extra_train_root"], mode)
        # Guard against leakage: extra data must not contain images or lesions of
        # the main dataset, whose val/test parts have to stay unseen.
        overlap = set(extra["image_id"]) & set(df["image_id"])
        overlap |= set(extra["lesion_id"]) & set(df["lesion_id"])
        if overlap:
            raise ValueError(
                f"Extra training data overlaps with the main dataset ({len(overlap)} ids, "
                f"e.g. {sorted(overlap)[:3]}). This would leak val/test images into training."
            )
        splits["train"] = pd.concat([splits["train"], extra], ignore_index=True)
    return splits, get_class_names(mode)

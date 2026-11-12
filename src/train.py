"""Train a skin lesion classifier.

Usage:
    python -m src.train --config configs/default.yaml
    python -m src.train --override task.mode=multiclass model.name=resnet50

Every run gets its own folder below ``runs/`` containing the exact config, the
split assignment, TensorBoard logs and the best checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.dataset import build_dataloaders, compute_class_weights, prepare_data
from src.evaluate import predict_loader
from src.model import build_model
from src.utils import compute_metrics, get_device, load_config, save_config, set_seed


def build_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """Linear warmup followed by cosine decay to zero (stepped once per batch)."""

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, device, cfg) -> float:
    model.train()
    amp = scaler.is_enabled()
    grad_clip = cfg["train"]["grad_clip"]
    total_loss, seen = 0.0, 0
    for images, targets in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        # Mixed precision: forward pass in float16 where safe -> less VRAM, faster.
        with torch.autocast(device_type=device.type, enabled=amp):
            loss = criterion(model(images), targets)
        scaler.scale(loss).backward()
        if grad_clip:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item() * images.size(0)
        seen += images.size(0)
    return total_loss / max(seen, 1)


def _scalars(metrics: dict) -> dict[str, float]:
    """Top-level numeric metrics (for TensorBoard)."""
    return {k: v for k, v in metrics.items() if isinstance(v, float) and not math.isnan(v)}


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default=None, help="YAML file merged over configs/default.yaml")
    parser.add_argument("--override", nargs="*", default=[], help="e.g. train.epochs=5")
    parser.add_argument("--device", default=None, help="cuda or cpu (default: auto)")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, args.override)
    set_seed(cfg["seed"])
    device = get_device(args.device)
    mode = cfg["task"]["mode"]

    # ---- run directory: everything needed to reproduce the run lives here
    run_name = cfg["logging"]["run_name"] or (
        f"{datetime.now():%Y%m%d-%H%M%S}_{cfg['model']['name']}_{mode}"
    )
    run_dir = Path(cfg["logging"]["runs_dir"]) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.yaml")

    # ---- data
    splits, class_names = prepare_data(cfg)
    pd.concat(
        [part[["image_id", "lesion_id", "dx", "label"]].assign(split=name)
         for name, part in splits.items()]
    ).to_csv(run_dir / "splits.csv", index=False)
    loaders = build_dataloaders(cfg, splits)
    print(f"Device: {device} | mode: {mode} | classes: {class_names}")
    print("Images per split: " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))

    # ---- model, loss, optimiser
    model = build_model(cfg, num_classes=len(class_names)).to(device)
    class_weights = None
    if cfg["train"]["imbalance"] == "weighted_loss":
        class_weights = compute_class_weights(
            splits["train"]["label"].to_numpy(), len(class_names)
        ).to(device)
        print(f"Class weights: {[round(w, 3) for w in class_weights.tolist()]}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"]
    )
    steps_per_epoch = len(loaders["train"])
    scheduler = build_scheduler(
        optimizer,
        warmup_steps=cfg["train"]["warmup_epochs"] * steps_per_epoch,
        total_steps=cfg["train"]["epochs"] * steps_per_epoch,
    )
    amp = bool(cfg["train"]["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    # ---- logging
    writer = SummaryWriter(run_dir / "tensorboard")
    wandb_run = None
    if cfg["logging"]["wandb"]:
        import wandb  # optional dependency

        wandb_run = wandb.init(project=cfg["logging"]["wandb_project"], name=run_name, config=cfg)

    monitor = cfg["train"]["monitor"]
    patience = cfg["train"]["early_stopping_patience"]
    best_score, best_epoch = -math.inf, 0

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        start = time.time()
        train_loss = train_one_epoch(
            model, loaders["train"], criterion, optimizer, scheduler, scaler, device, cfg
        )
        val_loss, y_true, y_prob = predict_loader(model, loaders["val"], device, amp)
        val = compute_metrics(y_true, y_prob, class_names, mode, cfg["task"]["threshold"])

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
        for key, value in _scalars(val).items():
            writer.add_scalar(f"val/{key}", value, epoch)
        if wandb_run is not None:
            wandb_run.log(
                {"epoch": epoch, "loss/train": train_loss, "loss/val": val_loss}
                | {f"val/{k}": v for k, v in _scalars(val).items()}
            )

        score = val[monitor]
        score = -math.inf if math.isnan(score) else score
        improved = score > best_score
        print(
            f"Epoch {epoch:3d}/{cfg['train']['epochs']} | train loss {train_loss:.4f} | "
            f"val loss {val_loss:.4f} | val auc {val['auc']:.4f} | "
            f"val recall {val['recall']:.4f} | {time.time() - start:.0f}s"
            + (" | * new best" if improved else "")
        )

        if improved:
            best_score, best_epoch = score, epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": cfg,
                    "class_names": class_names,
                    "epoch": epoch,
                    "monitor": monitor,
                    "best_score": float(score),
                },
                run_dir / "best.pt",
            )
            with open(run_dir / "best_val_metrics.json", "w", encoding="utf-8") as f:
                json.dump({"epoch": epoch, "loss": val_loss, **val}, f, indent=2)
        elif patience and epoch - best_epoch >= patience:
            print(f"Early stopping: no improvement in val {monitor} for {patience} epochs.")
            break

    writer.close()
    if wandb_run is not None:
        wandb_run.finish()
    print(f"\nBest val {monitor}: {best_score:.4f} (epoch {best_epoch})")
    print(f"Checkpoint: {run_dir / 'best.pt'}")
    print(f"Evaluate with: python -m src.evaluate --checkpoint {run_dir / 'best.pt'}")
    return run_dir


if __name__ == "__main__":
    main()

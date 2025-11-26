"""Utility helpers for smile classifier training."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import nn


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    best_metric: float,
    checkpoint_path: str | Path,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_metric": best_metric,
    }
    if scheduler is not None:
        checkpoint["scheduler_state"] = scheduler.state_dict()
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(
    checkpoint_path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    device: torch.device | str = "cpu",
) -> Tuple[int, float]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scheduler is not None and "scheduler_state" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    epoch = int(checkpoint.get("epoch", 0))
    best_metric = float(checkpoint.get("best_metric", 0.0))
    return epoch, best_metric


def classification_metrics(
    targets: np.ndarray,
    preds: np.ndarray,
) -> Dict[str, float]:
    accuracy = accuracy_score(targets, preds)
    precision = precision_score(targets, preds, zero_division=0)
    recall = recall_score(targets, preds, zero_division=0)
    f1 = f1_score(targets, preds, zero_division=0)
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }

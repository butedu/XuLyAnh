"""Training script for the smile classifier."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.models.smile_cnn import MODEL_IMAGE_SIZE, build_model
from src.training.datasets import DataLoaderConfig, make_dataloaders
from src.training.utils import classification_metrics, save_checkpoint, set_seed


def build_transforms(image_size: Tuple[int, int] = MODEL_IMAGE_SIZE) -> Dict[str, transforms.Compose]:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return {"train": train_transform, "val": eval_transform, "test": eval_transform}


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    for inputs, targets in tqdm(loader, desc="train", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(loader.dataset)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    running_loss = 0.0
    preds, targets_all = [], []
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="eval", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            loss = criterion(logits, targets)
            running_loss += loss.item() * inputs.size(0)
            predictions = torch.argmax(logits, dim=1)
            preds.extend(predictions.cpu().numpy())
            targets_all.extend(targets.cpu().numpy())
    avg_loss = running_loss / len(loader.dataset)
    metrics = classification_metrics(np.array(targets_all), np.array(preds))
    return avg_loss, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train smile classifier")
    parser.add_argument("--image-root", type=Path, required=True, help="Directory containing training images")
    parser.add_argument("--split-dir", type=Path, required=True, help="Directory with train/val/test CSVs")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models"),
        help="Directory to save checkpoints and final model",
    )
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--step-size", type=int, default=10, help="Scheduler step size")
    parser.add_argument("--gamma", type=float, default=0.5, help="Scheduler decay factor")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def run_training(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    transforms_dict = build_transforms()
    dataloaders = make_dataloaders(
        image_root=args.image_root,
        split_dir=args.split_dir,
        transform=transforms_dict["train"],
        config=DataLoaderConfig(batch_size=args.batch_size, num_workers=args.num_workers),
    )
    # Replace transforms for non-train loaders with evaluation transform
    eval_transform = transforms_dict["val"]
    for split in ("val", "test"):
        if split in dataloaders:
            dataset = dataloaders[split].dataset
            dataset.transform = eval_transform

    device = torch.device(args.device)
    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_loss = train_one_epoch(model, dataloaders["train"], criterion, optimizer, device)
        val_loss, val_metrics = evaluate(model, dataloaders["val"], criterion, device)
        scheduler.step()
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(record)
        print(json.dumps(record, indent=2))

        current_f1 = val_metrics.get("f1", 0.0)
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_path = output_dir / "smile_cnn_best.pth"
            torch.save(model.state_dict(), best_path)
            save_checkpoint(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                best_metric=best_f1,
                checkpoint_path=output_dir / "checkpoint.pt",
            )
            print(f"Saved new best model to {best_path}")

    history_path = output_dir / "training_history.json"
    with history_path.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    print(f"Training complete. History saved to {history_path}")


if __name__ == "__main__":
    run_training(parse_args())

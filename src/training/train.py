"""Huấn luyện bộ phân loại nụ cười."""

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

from src.classifier.smile_model import IMAGE_SIZE, build_model
from src.training.data import LoaderConfig, build_loaders
from src.training.utils import classification_report, save_checkpoint, set_seed


def build_transforms(image_size: Tuple[int, int] = IMAGE_SIZE) -> Dict[str, transforms.Compose]:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_tf = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return {"train": train_tf, "eval": eval_tf}


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    loss_sum = 0.0
    for images, labels in tqdm(loader, desc="train", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * images.size(0)
    return loss_sum / len(loader.dataset)


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, Dict[str, float]]:
    model.eval()
    loss_sum = 0.0
    preds: list[int] = []
    targets: list[int] = []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="eval", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            loss_sum += loss.item() * images.size(0)
            pred = torch.argmax(logits, dim=1)
            preds.extend(pred.cpu().numpy().tolist())
            targets.extend(labels.cpu().numpy().tolist())
    metrics = classification_report(np.array(targets), np.array(preds))
    return loss_sum / len(loader.dataset), metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Huấn luyện CNN phân loại nụ cười")
    parser.add_argument("--image-root", type=Path, required=True, help="Thư mục chứa ảnh gốc")
    parser.add_argument("--split-dir", type=Path, required=True, help="Thư mục có train/val/test CSV")
    parser.add_argument("--output-dir", type=Path, default=Path("models"), help="Nơi lưu checkpoint")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--step-size", type=int, default=10, help="Số epoch giảm lr")
    parser.add_argument("--gamma", type=float, default=0.5, help="Hệ số giảm lr")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def run_training(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(args.device)
    tfs = build_transforms()
    loaders = build_loaders(
        image_root=args.image_root,
        split_dir=args.split_dir,
        transform=tfs["train"],
        config=LoaderConfig(batch_size=args.batch_size, num_workers=args.num_workers),
    )
    for name in ("train", "val"):
        if name not in loaders:
            raise ValueError(f"Thiếu tập {name}. Hãy tạo file {name}.csv trong {args.split_dir}")
    eval_tf = tfs["eval"]
    for split in ("val", "test"):
        if split in loaders:
            dataset = loaders[split].dataset
            if hasattr(dataset, "transform"):
                setattr(dataset, "transform", eval_tf)

    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = 0.0
    history: list[Dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_loss = train_one_epoch(model, loaders["train"], criterion, optimizer, device)
        val_loss, val_metrics = evaluate(model, loaders["val"], criterion, device)
        scheduler.step()
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
        )
        print(json.dumps(history[-1], ensure_ascii=False, indent=2))
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
                path=output_dir / "checkpoint.pt",
            )
            print(f"Đã lưu mô hình tốt nhất vào {best_path}")

    history_path = output_dir / "training_history.json"
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print(f"Hoàn tất huấn luyện. Đã lưu lịch sử tại {history_path}")


def main() -> None:
    run_training(parse_args())


if __name__ == "__main__":
    main()

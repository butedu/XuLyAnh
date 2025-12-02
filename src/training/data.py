"""Chuẩn bị dữ liệu cho huấn luyện."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable

import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

Transform = Callable[[Image.Image], torch.Tensor]


class SmileDataset(Dataset):
    """Dataset đọc từ CSV với cột filepath,label."""

    def __init__(self, csv_path: str | Path, image_root: str | Path, transform: Transform) -> None:
        self.df = pd.read_csv(csv_path)
        if {"filepath", "label"} - set(self.df.columns):
            raise ValueError("CSV cần cột filepath và label")
        self.image_root = Path(image_root)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        path = self.image_root / row["filepath"]
        if not path.exists():
            raise FileNotFoundError(f"Thiếu ảnh: {path}")
        image = Image.open(path).convert("RGB")
        tensor = self.transform(image)
        label = int(row["label"])
        return tensor, label


@dataclass(slots=True)
class SplitConfig:
    train: float = 0.8
    val: float = 0.1
    test: float = 0.1
    seed: int = 42
    stratify: bool = True

    def validate(self) -> None:
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-6:
            raise ValueError("Tỷ lệ phải cộng thành 1.0")


def split_dataframe(df: pd.DataFrame, config: SplitConfig) -> Dict[str, pd.DataFrame]:
    config.validate()
    strat = df["label"] if config.stratify else None
    train_df, temp_df = train_test_split(
        df,
        train_size=config.train,
        random_state=config.seed,
        stratify=strat,
    )
    if config.test == 0:
        return {"train": train_df.reset_index(drop=True), "val": temp_df.reset_index(drop=True)}
    remain = config.val + config.test
    val_ratio = config.val / remain
    strat_temp = temp_df["label"] if config.stratify else None
    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_ratio,
        random_state=config.seed,
        stratify=strat_temp,
    )
    return {
        "train": train_df.reset_index(drop=True),
        "val": val_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
    }


def save_splits(splits: Dict[str, pd.DataFrame], output_dir: str | Path) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, subset in splits.items():
        subset.to_csv(out_dir / f"{name}.csv", index=False)


@dataclass(slots=True)
class LoaderConfig:
    batch_size: int = 64
    num_workers: int = 4
    pin_memory: bool = True


def build_loaders(image_root: str | Path, split_dir: str | Path, transform: Transform, config: LoaderConfig | None = None) -> Dict[str, DataLoader]:
    cfg = config or LoaderConfig()
    loaders: Dict[str, DataLoader] = {}
    for split in ("train", "val", "test"):
        csv_path = Path(split_dir) / f"{split}.csv"
        if not csv_path.exists():
            continue
        dataset = SmileDataset(csv_path, image_root, transform)
        loaders[split] = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=(split == "train"),
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
        )
    return loaders

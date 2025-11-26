"""Dataset utilities for smile classification training."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


Transform = Callable[[Image.Image], torch.Tensor]


class SmileImageDataset(Dataset):
    """Generic dataset that reads file paths and labels from a CSV file."""

    def __init__(
        self,
        annotations_file: str | Path,
        image_root: str | Path,
        transform: Transform,
    ) -> None:
        self.annotations = pd.read_csv(annotations_file)
        expected_columns = {"filepath", "label"}
        if not expected_columns.issubset(self.annotations.columns):
            raise ValueError(
                f"Annotation file must contain columns {expected_columns} but found {self.annotations.columns.tolist()}"
            )
        self.image_root = Path(image_root)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int):
        row = self.annotations.iloc[idx]
        image_path = self.image_root / row["filepath"]
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image)
        label = int(row["label"])
        return tensor, label


@dataclass(slots=True)
class DatasetSplitConfig:
    train_size: float = 0.8
    val_size: float = 0.1
    test_size: float = 0.1
    random_state: int = 42
    stratify: bool = True

    def validate(self) -> None:
        total = self.train_size + self.val_size + self.test_size
        if not abs(total - 1.0) < 1e-6:
            raise ValueError("Train/val/test ratios must sum to 1.0")


def create_splits(
    annotations: pd.DataFrame,
    config: DatasetSplitConfig,
) -> Dict[str, pd.DataFrame]:
    config.validate()
    stratify = annotations["label"] if config.stratify else None
    train_df, temp_df = train_test_split(
        annotations,
        train_size=config.train_size,
        random_state=config.random_state,
        stratify=stratify,
    )
    if config.test_size == 0:
        return {"train": train_df.reset_index(drop=True), "val": temp_df.reset_index(drop=True)}
    remaining = config.val_size + config.test_size
    val_relative = config.val_size / remaining
    stratify_temp = temp_df["label"] if config.stratify else None
    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_relative,
        random_state=config.random_state,
        stratify=stratify_temp,
    )
    return {
        "train": train_df.reset_index(drop=True),
        "val": val_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
    }


def save_splits(splits: Dict[str, pd.DataFrame], output_dir: str | Path) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    for split, df in splits.items():
        df.to_csv(output_path / f"{split}.csv", index=False)


@dataclass(slots=True)
class DataLoaderConfig:
    batch_size: int = 64
    num_workers: int = 4
    pin_memory: bool = True


def make_dataloaders(
    image_root: str | Path,
    split_dir: str | Path,
    transform: Transform,
    config: DataLoaderConfig | None = None,
) -> Dict[str, DataLoader]:
    config = config or DataLoaderConfig()
    loaders: Dict[str, DataLoader] = {}
    for split in ("train", "val", "test"):
        csv_path = Path(split_dir) / f"{split}.csv"
        if not csv_path.exists():
            continue
        dataset = SmileImageDataset(csv_path, image_root=image_root, transform=transform)
        shuffle = split == "train"
        loaders[split] = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=shuffle,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
        )
    return loaders

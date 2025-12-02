"""Compatibility wrapper for the legacy ``training.datasets`` module."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import pandas as pd
from torch.utils.data import DataLoader

from .data import LoaderConfig as _LoaderConfig
from .data import SmileDataset as SmileImageDataset
from .data import SplitConfig as _SplitConfig
from .data import Transform
from .data import build_loaders as _build_loaders
from .data import save_splits as _save_splits
from .data import split_dataframe as _split_dataframe
DataLoaderConfig = _LoaderConfig


@dataclass(slots=True)
class DatasetSplitConfig:
    train_size: float = 0.8
    val_size: float = 0.1
    test_size: float = 0.1
    random_state: int = 42
    stratify: bool = True

    def to_new_config(self) -> _SplitConfig:
        return _SplitConfig(
            train=self.train_size,
            val=self.val_size,
            test=self.test_size,
            seed=self.random_state,
            stratify=self.stratify,
        )


def create_splits(df: pd.DataFrame, config: DatasetSplitConfig | None = None) -> Dict[str, pd.DataFrame]:
    cfg = config or DatasetSplitConfig()
    return _split_dataframe(df, cfg.to_new_config())

def save_splits(splits: Dict[str, pd.DataFrame], output_dir: str | Path) -> None:
    _save_splits(splits, output_dir)


def make_dataloaders(
    image_root: str | Path,
    split_dir: str | Path,
    transform: Transform,
    config: DataLoaderConfig | None = None,
) -> Dict[str, DataLoader]:
    return _build_loaders(image_root, split_dir, transform, config)


__all__ = [
    "DataLoaderConfig",
    "DatasetSplitConfig",
    "SmileImageDataset",
    "Transform",
    "create_splits",
    "save_splits",
    "make_dataloaders",
]

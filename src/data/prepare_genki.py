"""Chuẩn hóa dữ liệu GENKI-4K: chuyển nhãn thành CSV và tạo các tập train/val/test."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd

from src.training.datasets import DatasetSplitConfig, create_splits, save_splits


def parse_labels(label_file: Path, image_subdir: str = "files") -> Iterable[Tuple[str, int]]:
    """Đọc labels.txt và gán tuần tự tới file ảnh fileXXXX.jpg tương ứng."""

    with label_file.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter=" ")
        for index, row in enumerate(reader, start=1):
            row = [token for token in row if token]
            if not row:
                continue
            label_token = row[0].strip()
            label = 1 if label_token == "1" else 0
            filename = f"{image_subdir}/file{index:04d}.jpg"
            yield filename, label


def build_annotations(raw_root: Path, output_csv: Path) -> pd.DataFrame:
    labels_path = raw_root / "labels.txt"
    if not labels_path.exists():
        raise FileNotFoundError(f"Không tìm thấy labels.txt tại {labels_path}")
    entries = list(parse_labels(labels_path))
    df = pd.DataFrame(entries, columns=["filepath", "label"])
    df.to_csv(output_csv, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Chuẩn bị dữ liệu GENKI-4K")
    parser.add_argument("--raw-dir", type=Path, required=True, help="Thư mục chứa ảnh và labels.txt của GENKI-4K")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/genki4k"),
        help="Thư mục lưu các file CSV kết quả",
    )
    parser.add_argument(
        "--create-splits",
        action="store_true",
        help="Tạo thêm các file train/val/test",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Tỉ lệ train nếu tạo splits",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Tỉ lệ validation",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Tỉ lệ test",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    annotations_csv = output_dir / "annotations.csv"
    df = build_annotations(args.raw_dir, annotations_csv)
    print(f"Đã lưu annotations tại {annotations_csv} ({len(df)} dòng)")

    if args.create_splits:
        split_config = DatasetSplitConfig(
            train_size=args.train_ratio,
            val_size=args.val_ratio,
            test_size=args.test_ratio,
        )
        splits = create_splits(df, split_config)
        save_splits(splits, output_dir)
        for split_name, split_df in splits.items():
            print(f"Đã lưu {split_name}.csv với {len(split_df)} dòng")


if __name__ == "__main__":
    main()

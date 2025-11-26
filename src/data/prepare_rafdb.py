"""Chuyển nhãn RAF-DB sang định dạng nhị phân cười/không cười."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import pandas as pd


EXPRESSION_TO_SMILE: Dict[int, int] = {
    1: 0,  # Surprise
    2: 0,  # Fear
    3: 0,  # Disgust
    4: 1,  # Happiness
    5: 0,  # Sadness
    6: 0,  # Anger
    7: 0,  # Neutral
}


def load_partition_labels(label_file: Path) -> pd.DataFrame:
    if not label_file.exists():
        raise FileNotFoundError(f"Không tìm thấy {label_file}")
    data = []
    with label_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            tokens = line.strip().split()
            if len(tokens) != 2:
                continue
            filename, label_str = tokens
            label = int(label_str)
            smile = EXPRESSION_TO_SMILE.get(label)
            if smile is None:
                continue
            data.append((filename, smile))
    return pd.DataFrame(data, columns=["filepath", "label"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Chuẩn bị dữ liệu RAF-DB")
    parser.add_argument("--raw-dir", type=Path, required=True, help="Thư mục gốc RAF-DB đã giải nén")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/rafdb"),
        help="Thư mục lưu các file CSV",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    label_root = args.raw_dir / "EmoLabel"
    train_labels = load_partition_labels(label_root / "list_patition_label.txt")
    train_split = train_labels[train_labels["filepath"].str.startswith("train")].reset_index(drop=True)
    test_split = train_labels[train_labels["filepath"].str.startswith("test")].reset_index(drop=True)

    train_split.to_csv(output_dir / "train.csv", index=False)
    test_split.to_csv(output_dir / "test.csv", index=False)

    annotations = pd.concat([train_split.assign(split="train"), test_split.assign(split="test")])
    annotations.to_csv(output_dir / "annotations.csv", index=False)

    print(f"Đã lưu RAF-DB annotations tại {output_dir}")
    print(f"Train: {len(train_split)}, Test: {len(test_split)}")


if __name__ == "__main__":
    main()

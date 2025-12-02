"""Trình bao giữ tương thích cho lệnh huấn luyện cũ."""

from __future__ import annotations

import argparse

from src.training.train import parse_args as _parse_args_new
from src.training.train import run_training as _run_training_new


def parse_args() -> argparse.Namespace:
    """Giữ API cũ nhưng in cảnh báo nhẹ."""

    print("[Cảnh báo] Lệnh mới là `python -m src.training.train`. Vẫn tiếp tục chạy...")
    return _parse_args_new()


def run_training(args: argparse.Namespace) -> None:
    _run_training_new(args)


if __name__ == "__main__":
    run_training(parse_args())

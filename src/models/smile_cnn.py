"""Giữ tương thích cho module cũ bằng cách chuyển sang SmileNet mới."""
from __future__ import annotations

from typing import Tuple

import torch

from src.classifier.smile_model import (
    IMAGE_SIZE as _IMAGE_SIZE,
    SmileNet as SmileCNN,
    SmileNetConfig as _SmileNetConfig,
    build_model as _build_model,
)

CauHinhMoHinhCuoi = _SmileNetConfig
MODEL_IMAGE_SIZE: Tuple[int, int] = _IMAGE_SIZE


def xay_dung_mo_hinh(config: CauHinhMoHinhCuoi | None = None) -> SmileCNN:
    return _build_model(config)


def build_model(config: CauHinhMoHinhCuoi | None = None) -> SmileCNN:
    return xay_dung_mo_hinh(config)


def tai_mo_hinh(duong_dan_weights: str, thiet_bi: torch.device | str = "cpu") -> SmileCNN:
    return load_model(duong_dan_weights, thiet_bi)


def load_model(weights_path: str, device: torch.device | str = "cpu") -> SmileCNN:
    model = xay_dung_mo_hinh()
    state = torch.load(weights_path, map_location=device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint không khớp: thiếu {missing}, thừa {unexpected}")
    return model.to(device)


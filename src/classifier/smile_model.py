"""Kiến trúc CNN nhẹ cho nhận diện nụ cười."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor, nn

IMAGE_SIZE: Tuple[int, int] = (64, 64)


class SmileNet(nn.Module):
    """Mạng CNN 2 lớp fully-connected cho bài toán nhị phân."""

    def __init__(self, in_channels: int = 3, dropout: float = 0.3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256 * 2 * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 2),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        """Khởi tạo trọng số nhanh."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.features(x))


@dataclass(slots=True)
class SmileNetConfig:
    """Cấu hình ngắn gọn."""

    in_channels: int = 3
    dropout: float = 0.3
    num_classes: int = 2

    def build(self) -> SmileNet:
        model = SmileNet(self.in_channels, self.dropout)
        if self.num_classes != 2:
            last = model.classifier[-1]
            if not isinstance(last, nn.Linear):  # pragma: no cover
                raise TypeError("Lớp cuối không phải Linear")
            model.classifier[-1] = nn.Linear(last.in_features, self.num_classes)
        return model


def build_model(config: SmileNetConfig | None = None) -> SmileNet:
    return (config or SmileNetConfig()).build()


def load_weights(weights_path: str, device: torch.device | str = "cpu") -> SmileNet:
    model = build_model()
    state = torch.load(weights_path, map_location=device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint không khớp: thiếu {missing}, thừa {unexpected}")
    return model.to(device)

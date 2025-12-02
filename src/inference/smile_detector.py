"""Trình bao tương thích sử dụng pipeline SmileCounter mới."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from src.pipeline.smile_counter import SmileCounter, SmileCounterConfig


@dataclass(slots=True)
class CauHinhBoNhanDienCuoi(SmileCounterConfig):
    """Giữ tên cũ cho mã nguồn hiện có."""


class BoNhanDienCuoi(SmileCounter):
    """Bao quanh SmileCounter để tương thích với API cũ."""

    def __init__(self, config: CauHinhBoNhanDienCuoi | None = None) -> None:
        super().__init__(config or CauHinhBoNhanDienCuoi())

    def phan_tich(self, anh_bgr) -> Dict[str, object]:  # type: ignore[override]
        """Alias tiếng Việt cho `analyze_array`."""

        return self.analyze_array(anh_bgr)

    def chu_thich_anh(self, anh_bgr, detections: List[Dict[str, object]]):  # type: ignore[override]
        return self.annotate(anh_bgr, detections)

    def phan_tich_tu_file(self, duong_dan_anh: str | Path) -> Dict[str, object]:
        return self.analyze_file(duong_dan_anh)


# Alias tiếng Anh để tương thích
SmileDetectorConfig = CauHinhBoNhanDienCuoi
SmileDetector = BoNhanDienCuoi
        

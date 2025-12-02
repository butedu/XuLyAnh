"""Trình bao YOLO cho phát hiện khuôn mặt."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

try:  # Lazy import để báo lỗi rõ ràng
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover
    raise ImportError("Thiếu gói ultralytics, hãy cài 'pip install ultralytics'.") from exc


@dataclass(slots=True)
class YOLOFaceConfig:
    """Cấu hình cho bộ phát hiện mặt."""

    model_path: str | None = "models/yolov8n-face.pt"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    conf: float = 0.25
    iou: float = 0.45
    max_faces: int = 50


class YOLOFaceDetector:
    """Bộ phát hiện mặt dùng YOLOv8-face."""

    def __init__(self, config: YOLOFaceConfig | None = None) -> None:
        self.config = config or YOLOFaceConfig()
        if self.config.device.startswith("cuda") and not torch.cuda.is_available():
            self.config.device = "cpu"
        model_source = self._resolve_model_path(self.config.model_path)
        self.model = YOLO(model_source)

    @staticmethod
    def _resolve_model_path(model_path: str | None) -> str:
        """Chuẩn hóa đường dẫn model."""
        if model_path is None:
            return "yolov8n-face.pt"
        path = Path(model_path)
        if path.exists():
            return str(path)
        return model_path  # Cho phép dùng tên model có sẵn

    def detect(self, image_bgr: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """Trả về danh sách (x, y, w, h, score)."""
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("Ảnh đầu vào trống")
        image_rgb = image_bgr[..., ::-1]
        result = self.model(
            image_rgb,
            imgsz=640,
            conf=self.config.conf,
            iou=self.config.iou,
            device=self.config.device,
            max_det=self.config.max_faces,
        )[0]
        boxes: List[Tuple[int, int, int, int, float]] = []
        if not hasattr(result, "boxes"):
            return boxes
        xyxy = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), score in zip(xyxy, scores):
            x = int(x1)
            y = int(y1)
            w = int(max(0, x2 - x1))
            h = int(max(0, y2 - y1))
            boxes.append((x, y, w, h, float(score)))
        return boxes

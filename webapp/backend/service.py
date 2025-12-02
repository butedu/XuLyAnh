"""Dịch vụ FastAPI sử dụng pipeline mới."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import torch

from src.pipeline.smile_counter import SmileCounter, SmileCounterConfig


class DichVuNhanDienCuoi:
    """Bao lớp SmileCounter cho web API."""

    def __init__(
        self,
        duong_dan_mo_hinh: str | Path | None = None,
        duong_dan_face: str | Path | None = "models/yolov8n-face.pt",
        device: str | None = None,
    ) -> None:
        config = SmileCounterConfig(
            classifier_weights=str(duong_dan_mo_hinh or "models/smile_cnn_best.pth"),
            face_model=str(duong_dan_face) if duong_dan_face is not None else None,
            device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
        )
        self.pipeline = SmileCounter(config)

    def phan_tich_anh_bytes(self, anh_bytes: bytes) -> Dict[str, object]:
        arr = np.frombuffer(anh_bytes, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Tệp tải lên không phải ảnh hợp lệ")
        return self.pipeline.analyze_array(image)

    def chu_thich_anh(self, anh_bytes: bytes, ket_qua: Dict[str, object]) -> bytes:
        arr = np.frombuffer(anh_bytes, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Tệp tải lên không phải ảnh hợp lệ")
        annotated = self.pipeline.annotate(image, ket_qua.get("detections", []))
        ok, buffer = cv2.imencode(".jpg", annotated)
        if not ok:
            raise ValueError("Không thể mã hóa ảnh")
        return bytes(buffer)


# Alias giữ tương thích
SmileService = DichVuNhanDienCuoi
analyze_image_bytes = DichVuNhanDienCuoi.phan_tich_anh_bytes
annotate_image = DichVuNhanDienCuoi.chu_thich_anh

"""Lớp dịch vụ hỗ trợ FastAPI."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import cv2
import numpy as np

from src.inference.smile_detector import SmileDetector, SmileDetectorConfig


class DichVuNhanDienCuoi:
    def __init__(self, duong_dan_mo_hinh: str | Path | None = None) -> None:
        config = SmileDetectorConfig()
        if duong_dan_mo_hinh is not None:
            config.model_weights = str(duong_dan_mo_hinh)
        self.detector = SmileDetector(config)

    def phan_tich_anh_bytes(self, anh_bytes: bytes) -> Dict[str, object]:
        du_lieu_np = np.frombuffer(anh_bytes, dtype=np.uint8)
        anh_bgr = cv2.imdecode(du_lieu_np, cv2.IMREAD_COLOR)
        if anh_bgr is None:
            raise ValueError("Tệp tải lên không phải ảnh hợp lệ")
        return self.detector.phan_tich(anh_bgr)

    def chu_thich_anh(self, anh_bytes: bytes, ket_qua: Dict[str, object]) -> bytes:
        du_lieu_np = np.frombuffer(anh_bytes, dtype=np.uint8)
        anh_bgr = cv2.imdecode(du_lieu_np, cv2.IMREAD_COLOR)
        if anh_bgr is None:
            raise ValueError("Tệp tải lên không phải ảnh hợp lệ")
        boxes = ket_qua.get("detections", [])
        anh_chu_thich = self.detector.chu_thich_anh(anh_bgr, boxes)
        _, encoded = cv2.imencode(".jpg", anh_chu_thich)
        return bytes(encoded)


# Alias tiếng Anh để tương thích với các đoạn mã cũ nếu cần
SmileService = DichVuNhanDienCuoi
analyze_image_bytes = DichVuNhanDienCuoi.phan_tich_anh_bytes
annotate_image = DichVuNhanDienCuoi.chu_thich_anh

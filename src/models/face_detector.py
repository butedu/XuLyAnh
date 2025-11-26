"""Các hàm phát hiện khuôn mặt sử dụng Haar cascades hoặc DNN của OpenCV."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Sequence, Tuple

import cv2
import numpy as np

# Kiểu backend cho detector: "haar" (nhanh) hoặc "dnn" (chính xác hơn)
KieuBoNhanDien = Literal["haar", "dnn"]


@dataclass(slots=True)
class CauHinhBoNhanDienKhuonMat:
    """Cấu hình cho bộ phát hiện khuôn mặt.
    
    Thuộc tính:
        backend: Loại detector ('haar' hoặc 'dnn')
        min_confidence: Ngưỡng confidence tối thiểu cho DNN (0-1)
        scale_factor: Hệ số scale cho Haar cascade
        min_neighbors: Số neighbors tối thiểu cho Haar cascade
        model_base_path: Đường dẫn đến thư mục chứa model DNN
    """
    backend: KieuBoNhanDien = "haar"
    min_confidence: float = 0.5
    scale_factor: float = 1.1
    min_neighbors: int = 5
    model_base_path: str | None = None

    def lay_duong_dan_model(self) -> Tuple[Path | None, Path | None]:
        """Lấy đường dẫn đến các file model DNN."""
        if self.backend != "dnn":
            return None, None
        base = Path(self.model_base_path or "models/face_detector")
        proto = base / "deploy.prototxt"
        weights = base / "res10_300x300_ssd_iter_140000.caffemodel"
        return proto, weights


class BoNhanDienKhuonMat:
    """Wrapper nhẹ bao quanh các bộ phát hiện khuôn mặt của OpenCV.
    
    Hỗ trợ 2 phương pháp:
    - Haar Cascade: Nhanh nhưng độ chính xác thấp hơn
    - DNN (Deep Neural Network): Chậm hơn nhưng chính xác hơn
    """

    def __init__(self, config: CauHinhBoNhanDienKhuonMat | None = None) -> None:
        """Khởi tạo bộ phát hiện khuôn mặt.
        
        Tham số:
            config: Cấu hình detector (nếu None sẽ dùng mặc định)
        """
        self.config = config or CauHinhBoNhanDienKhuonMat()
        if self.config.backend == "haar":
            # Tải Haar Cascade từ OpenCV
            cascade_path = str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
            self.detector = cv2.CascadeClassifier(cascade_path)
            if self.detector.empty():
                raise RuntimeError(f"Không thể tải Haar cascade từ {cascade_path}")
        elif self.config.backend == "dnn":
            # Tải DNN model từ file
            proto, weights = self.config.lay_duong_dan_model()
            if not proto or not proto.exists() or not weights or not weights.exists():
                raise FileNotFoundError(
                    f"Thiếu file DNN face detector. Cần có deploy.prototxt và .caffemodel trong "
                    f"{self.config.model_base_path or 'models/face_detector'}"
                )
            self.detector = cv2.dnn.readNetFromCaffe(str(proto), str(weights))
        else:
            raise ValueError(f"Backend không được hỗ trợ: {self.config.backend}")

    def phat_hien(self, anh_bgr: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """Phát hiện khuôn mặt trong ảnh.
        
        Tham số:
            anh_bgr: Ảnh đầu vào ở định dạng BGR (OpenCV)
            
        Trả về:
            Danh sách các khuôn mặt, mỗi phần tử là tuple (x, y, w, h, confidence)
            - x, y: Tọa độ góc trên bên trái
            - w, h: Chiều rộng và chiều cao
            - confidence: Độ tin cậy (0-1)
        """

        if self.config.backend == "haar":
            # Phát hiện bằng Haar Cascade
            gray = cv2.cvtColor(anh_bgr, cv2.COLOR_BGR2GRAY)
            boxes = self.detector.detectMultiScale(
                gray,
                scaleFactor=self.config.scale_factor,
                minNeighbors=self.config.min_neighbors,
            )
            ket_qua: List[Tuple[int, int, int, int, float]] = []
            for (x, y, w, h) in boxes:
                ket_qua.append((int(x), int(y), int(w), int(h), 1.0))
            return ket_qua

        # Phát hiện bằng DNN
        height, width = anh_bgr.shape[:2]
        # Chuẩn bị blob cho DNN (resize về 300x300, chuẩn hóa)
        blob = cv2.dnn.blobFromImage(
            cv2.resize(anh_bgr, (300, 300)),
            scalefactor=1.0,
            size=(300, 300),
            mean=(104.0, 177.0, 123.0),  # Mean subtraction
        )
        self.detector.setInput(blob)
        detections = self.detector.forward()
        ket_qua = []
        # Duyệt qua các detection
        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence < self.config.min_confidence:
                continue
            # Chuyển tọa độ về pixel gốc
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            w, h = max(0, x2 - x1), max(0, y2 - y1)
            ket_qua.append((x1, y1, w, h, confidence))
        return ket_qua

    @staticmethod
    def mo_rong_vung(
        boxes: Sequence[Tuple[int, int, int, int, float]],
        kich_thuoc_anh: Tuple[int, int, int],
        ty_le: float = 1.2,
    ) -> List[Tuple[int, int, int, int, float]]:
        """Mở rộng các bounding box để bắt thêm ngữ cảnh xung quanh khuôn mặt.
        
        Tham số:
            boxes: Danh sách các box (x, y, w, h, confidence)
            kich_thuoc_anh: Shape của ảnh (height, width, channels)
            ty_le: Hệ số mở rộng (1.2 = tăng 20% mỗi chiều)
            
        Trả về:
            Danh sách các box đã được mở rộng
        """
        height, width = kich_thuoc_anh[:2]
        boxes_mo_rong: List[Tuple[int, int, int, int, float]] = []
        for x, y, w, h, conf in boxes:
            # Tính tâm box
            cx, cy = x + w / 2.0, y + h / 2.0
            # Tính kích thước mới
            new_w, new_h = w * ty_le, h * ty_le
            # Tính lại tọa độ (đảm bảo không vượt biên)
            nx = int(max(0, cx - new_w / 2.0))
            ny = int(max(0, cy - new_h / 2.0))
            nw = int(min(width - nx, new_w))
            nh = int(min(height - ny, new_h))
            boxes_mo_rong.append((nx, ny, nw, nh, conf))
        return boxes_mo_rong

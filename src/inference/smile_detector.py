"""Pipeline suy luận cho phát hiện nụ cười trong ảnh nhóm."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

from src.models.face_detector import BoNhanDienKhuonMat, CauHinhBoNhanDienKhuonMat
from src.models.smile_cnn import MODEL_IMAGE_SIZE, xay_dung_mo_hinh


@dataclass(slots=True)
class CauHinhBoNhanDienCuoi:
    """Cấu hình cho bộ phát hiện nụ cười.
    
    Thuộc tính:
        face_backend: Loại detector khuôn mặt ('haar' hoặc 'dnn')
        face_min_confidence: Ngưỡng confidence tối thiểu cho face detection
        model_weights: Đường dẫn đến file trọng số mô hình CNN
        device: Thiết bị chạy mô hình ('cpu' hoặc 'cuda')
    """
    face_backend: str = "haar"
    face_min_confidence: float = 0.5
    model_weights: str = "models/smile_cnn_best.pth"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class BoNhanDienCuoi:
    """Bộ phát hiện nụ cười kết hợp face detection và smile classification.
    
    Pipeline:
    1. Phát hiện khuôn mặt trong ảnh
    2. Cắt từng khuôn mặt
    3. Phân loại cười/không cười
    4. Thống kê kết quả
    """
    
    def __init__(self, config: CauHinhBoNhanDienCuoi | None = None) -> None:
        """Khởi tạo bộ phát hiện nụ cười.
        
        Tham số:
            config: Cấu hình (nếu None sẽ dùng mặc định)
        """
        self.config = config or CauHinhBoNhanDienCuoi()
        # Khởi tạo bộ phát hiện khuôn mặt
        face_config = CauHinhBoNhanDienKhuonMat(
            backend=self.config.face_backend,
            min_confidence=self.config.face_min_confidence,
        )
        self.face_detector = BoNhanDienKhuonMat(face_config)
        self.device = torch.device(self.config.device)
        # Tải mô hình phân loại nụ cười
        self.model = self._tai_mo_hinh(self.config.model_weights)
        # Chuẩn bị transform cho ảnh đầu vào
        self.transform = transforms.Compose(
            [
                transforms.Resize(MODEL_IMAGE_SIZE),  # Resize về 64x64
                transforms.ToTensor(),  # Chuyển sang tensor
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Chuẩn hóa
            ]
        )

    def _tai_mo_hinh(self, duong_dan_weights: str | Path) -> torch.nn.Module:
        """Tải mô hình CNN từ file checkpoint."""
        weights = Path(duong_dan_weights)
        if not weights.exists():
            raise FileNotFoundError(
                f"Không tìm thấy file trọng số tại {weights}. "
                f"Vui lòng huấn luyện mô hình trước hoặc cập nhật đường dẫn."
            )
        model = xay_dung_mo_hinh()
        state_dict = torch.load(weights, map_location=self.device)
        model.load_state_dict(state_dict)
        model.eval()  # Chuyển sang chế độ đánh giá
        return model.to(self.device)

    def _chuan_bi_khuon_mat(self, anh: np.ndarray) -> torch.Tensor:
        """Chuẩn bị ảnh khuôn mặt cho mô hình.
        
        Tham số:
            anh: Ảnh BGR từ OpenCV
            
        Trả về:
            Tensor đã chuẩn bị cho mô hình
        """
        anh_rgb = cv2.cvtColor(anh, cv2.COLOR_BGR2RGB)
        tensor = self.transform(Image.fromarray(anh_rgb))
        return tensor.unsqueeze(0).to(self.device)  # Thêm batch dimension

    def phat_hien_khuon_mat(self, anh_bgr: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """Phát hiện tất cả khuôn mặt trong ảnh.
        
        Tham số:
            anh_bgr: Ảnh đầu vào BGR
            
        Trả về:
            Danh sách các khuôn mặt (x, y, w, h, confidence)
        """
        boxes = self.face_detector.phat_hien(anh_bgr)
        return BoNhanDienKhuonMat.mo_rong_vung(boxes, anh_bgr.shape)

    def du_doan_cuoi(self, anh_khuon_mat: np.ndarray) -> Tuple[int, float]:
        """Dự đoán xem khuôn mặt có đang cười không.
        
        Tham số:
            anh_khuon_mat: Ảnh khuôn mặt đã cắt BGR
            
        Trả về:
            (nhãn, xác suất): nhãn=1 nếu cười, nhãn=0 nếu không cười
        """
        input_tensor = self._chuan_bi_khuon_mat(anh_khuon_mat)
        with torch.no_grad():  # Không tính gradient
            logits = self.model(input_tensor)
            probs = torch.softmax(logits, dim=1)  # Chuyển sang xác suất
            xac_suat_cuoi = float(probs[0, 1].cpu().item())
        nhan = 1 if xac_suat_cuoi >= 0.5 else 0
        return nhan, xac_suat_cuoi

    def phan_tich(self, anh_bgr: np.ndarray) -> Dict[str, object]:
        """Phân tích ảnh và đếm số người cười.
        
        Tham số:
            anh_bgr: Ảnh đầu vào BGR
            
        Trả về:
            Dictionary chứa:
            - total_faces: Tổng số khuôn mặt
            - smiling_faces: Số người đang cười
            - detections: Danh sách chi tiết từng khuôn mặt
        """
        cac_phat_hien = []
        cac_khuon_mat = self.phat_hien_khuon_mat(anh_bgr)
        so_nguoi_cuoi = 0
        
        for x, y, w, h, conf in cac_khuon_mat:
            # Cắt khuôn mặt từ ảnh
            anh_khuon_mat = anh_bgr[y : y + h, x : x + w]
            if anh_khuon_mat.size == 0:
                continue
            
            # Dự đoán cười/không cười
            nhan, xac_suat = self.du_doan_cuoi(anh_khuon_mat)
            so_nguoi_cuoi += int(nhan)
            
            cac_phat_hien.append(
                {
                    "box": [int(x), int(y), int(w), int(h)],
                    "confidence": float(conf),
                    "is_smiling": bool(nhan),
                    "smile_probability": xac_suat,
                }
            )
        
        return {
            "total_faces": len(cac_khuon_mat),
            "smiling_faces": so_nguoi_cuoi,
            "detections": cac_phat_hien,
        }

    def chu_thich_anh(self, anh_bgr: np.ndarray, cac_phat_hien: List[Dict[str, object]]) -> np.ndarray:
        """Vẽ khung và nhãn lên ảnh.
        
        Tham số:
            anh_bgr: Ảnh gốc BGR
            cac_phat_hien: Danh sách các detection
            
        Trả về:
            Ảnh đã được chú thích
        """
        anh_chu_thich = anh_bgr.copy()
        for phat_hien in cac_phat_hien:
            x, y, w, h = phat_hien["box"]
            dang_cuoi = phat_hien["is_smiling"]
            # Màu xanh lá cho cười, đỏ cho không cười
            mau = (0, 255, 0) if dang_cuoi else (0, 0, 255)
            # Vẽ khung chữ nhật
            cv2.rectangle(anh_chu_thich, (x, y), (x + w, y + h), mau, 2)
            # Ghi nhãn
            nhan = "Cuoi" if dang_cuoi else "Binh thuong"
            xac_suat = phat_hien["smile_probability"]
            text = f"{nhan}: {xac_suat:.2f}"
            cv2.putText(anh_chu_thich, text, (x, max(0, y - 10)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, mau, 1)
        return anh_chu_thich

    def phan_tich_tu_file(self, duong_dan_anh: str | Path) -> Dict[str, object]:
        """Phân tích ảnh từ file.
        
        Tham số:
            duong_dan_anh: Đường dẫn đến file ảnh
            
        Trả về:
            Kết quả phân tích
        """
        anh = cv2.imread(str(duong_dan_anh))
        if anh is None:
            raise FileNotFoundError(f"Không thể đọc ảnh tại {duong_dan_anh}")
        ket_qua = self.phan_tich(anh)
        return ket_qua



















# Alias tiếng Anh để tương thích với các module cũ
SmileDetectorConfig = CauHinhBoNhanDienCuoi
SmileDetector = BoNhanDienCuoi

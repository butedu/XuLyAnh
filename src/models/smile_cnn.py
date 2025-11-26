"""Kiến trúc mạng CNN cho bài toán phân loại nụ cười.

Mạng này được thiết kế nhẹ để huấn luyện nhanh trên phần cứng thông thường,
đồng thời vẫn đủ mạnh cho các tập dữ liệu GENKI-4K và RAF-DB.
Kiến trúc sử dụng 4 khối tích chập theo sau bởi lớp fully connected nhỏ.
BatchNorm và Dropout được dùng để cải thiện độ ổn định và hạn chế overfitting.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor, nn


# Kích thước ảnh đầu vào cho mô hình (64x64 pixels)
MODEL_IMAGE_SIZE: Tuple[int, int] = (64, 64)


class SmileCNN(nn.Module):
    """Mạng CNN đơn giản để phân loại nhị phân: cười/không cười.
    
    Tham số:
        input_channels: Số kênh màu đầu vào (3 cho RGB)
        dropout: Tỷ lệ dropout để tránh overfitting (mặc định 0.35)
    """

    def __init__(self, input_channels: int = 3, dropout: float = 0.35) -> None:
        super().__init__()
        # Các lớp trích xuất đặc trưng (convolutional layers)
        self.features = nn.Sequential(
            # Khối 1: 3 -> 32 channels, giảm kích thước xuống 32x32
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),  # Lớp tích chập
            nn.BatchNorm2d(32),  # Chuẩn hóa batch
            nn.ReLU(inplace=True),  # Hàm kích hoạt
            nn.MaxPool2d(2),  # Giảm kích thước 1/2
            
            # Khối 2: 32 -> 64 channels, giảm xuống 16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Khối 3: 64 -> 128 channels, giảm xuống 8x8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Khối 4: 128 -> 256 channels, giảm xuống 2x2
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((2, 2)),  # Pooling thích ứng về kích thước cố định
        )
        # Lớp phân loại (fully connected layers)
        self.classifier = nn.Sequential(
            nn.Flatten(),  # Làm phẳng từ 256x2x2 thành vector 1024
            nn.Dropout(dropout),  # Dropout để tránh overfitting
            nn.Linear(256 * 2 * 2, 256),  # Lớp ẩn với 256 neurons
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 2),  # Lớp đầu ra: 2 classes (cười/không cười)
        )
        self._khoi_tao_trong_so()

    def _khoi_tao_trong_so(self) -> None:
        """Khởi tạo trọng số cho các lớp mạng."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                # Khởi tạo He cho Conv2D (tốt với ReLU)
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Linear):
                # Khởi tạo Xavier cho Linear
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, x: Tensor) -> Tensor:
        """Lan truyền xuôi qua mạng.
        
        Tham số:
            x: Tensor ảnh đầu vào shape (batch_size, 3, 64, 64)
            
        Trả về:
            Tensor logits shape (batch_size, 2)
        """
        x = self.features(x)  # Trích xuất đặc trưng
        return self.classifier(x)  # Phân loại


@dataclass(slots=True)
class CauHinhMoHinhCuoi:
    """Cấu hình để xây dựng mô hình phân loại nụ cười.
    
    Thuộc tính:
        input_channels: Số kênh đầu vào (3 cho RGB, 1 cho grayscale)
        dropout: Tỷ lệ dropout (0-1)
        num_classes: Số lớp phân loại (mặc định 2: cười/không cười)
    """

    input_channels: int = 3
    dropout: float = 0.35
    num_classes: int = 2

    def xay_dung(self) -> SmileCNN:
        """Xây dựng mô hình từ cấu hình."""
        model = SmileCNN(input_channels=self.input_channels, dropout=self.dropout)
        if self.num_classes != 2:
            # Thay thế lớp cuối nếu cần số lớp khác
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, self.num_classes)
        return model


def xay_dung_mo_hinh(config: CauHinhMoHinhCuoi | None = None) -> SmileCNN:
    """Hàm tiện ích để tạo mô hình phân loại.
    
    Tham số:
        config: Cấu hình mô hình (nếu None sẽ dùng cấu hình mặc định)
        
    Trả về:
        Mô hình SmileCNN đã được khởi tạo
    """
    config = config or CauHinhMoHinhCuoi()
    return config.xay_dung()


def tai_mo_hinh(duong_dan_weights: str, thiet_bi: torch.device | str = "cpu") -> SmileCNN:
    """Tải mô hình đã huấn luyện từ file checkpoint.
    
    Tham số:
        duong_dan_weights: Đường dẫn đến file .pth chứa trọng số
        thiet_bi: Thiết bị để tải mô hình lên ('cpu' hoặc 'cuda')
        
    Trả về:
        Mô hình đã được tải trọng số
        
    Lỗi:
        RuntimeError: Nếu checkpoint không khớp với kiến trúc mô hình
    """
    model = xay_dung_mo_hinh()
    state_dict = torch.load(duong_dan_weights, map_location=thiet_bi)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint không khớp: thiếu keys {missing}, thừa keys {unexpected}"
        )
    return model.to(thiet_bi)


# Giữ lại hàm với tên tiếng Anh để tương thích ngược với các module cũ.
def build_model(config: CauHinhMoHinhCuoi | None = None) -> SmileCNN:
    return xay_dung_mo_hinh(config)


def load_model(weights_path: str, device: torch.device | str = "cpu") -> SmileCNN:
    return tai_mo_hinh(weights_path, device)

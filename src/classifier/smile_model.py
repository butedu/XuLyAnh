"""Kiến trúc CNN nhẹ cho nhận diện nụ cười."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor, nn

IMAGE_SIZE: Tuple[int, int] = (64, 64)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block để tăng cường attention cho channels quan trọng."""
    
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: Tensor) -> Tensor:
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResidualBlock(nn.Module):
    """Residual block với BatchNorm và optional SE attention."""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, use_se: bool = True) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.se = SEBlock(out_channels) if use_se else nn.Identity()
        
        # Shortcut connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x: Tensor) -> Tensor:
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        
        out += identity
        out = self.relu(out)
        return out


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


class SmileNetV2(nn.Module):
    """Kiến trúc CNN cải tiến với Residual blocks và SE attention."""
    
    def __init__(self, in_channels: int = 3, dropout: float = 0.3, use_se: bool = True) -> None:
        super().__init__()
        
        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Residual blocks với tăng channels dần
        self.layer1 = self._make_layer(32, 64, num_blocks=2, stride=2, use_se=use_se)
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2, use_se=use_se)
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2, use_se=use_se)
        
        # Global pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),  # Dropout nhẹ hơn ở layer cuối
            nn.Linear(128, 2)
        )
        
        self._init_weights()
    
    def _make_layer(self, in_channels: int, out_channels: int, num_blocks: int, stride: int, use_se: bool) -> nn.Sequential:
        """Tạo một layer gồm nhiều residual blocks."""
        layers = []
        # Block đầu tiên có stride để downsampling
        layers.append(ResidualBlock(in_channels, out_channels, stride, use_se))
        # Các block sau giữ nguyên kích thước
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1, use_se))
        return nn.Sequential(*layers)
    
    def _init_weights(self) -> None:
        """Khởi tạo trọng số với Kaiming initialization cho Conv và Xavier cho Linear."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:  # Chỉ init nếu có bias
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:  # Chỉ init nếu có bias
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x


@dataclass(slots=True)
class SmileNetConfig:
    """Cấu hình ngắn gọn."""

    in_channels: int = 3
    dropout: float = 0.3
    num_classes: int = 2
    model_name: str = "SmileNet"  # "SmileNet" hoặc "SmileNetV2"
    use_se_block: bool = True  # Chỉ cho SmileNetV2

    def build(self) -> nn.Module:
        if self.model_name == "SmileNetV2":
            model = SmileNetV2(self.in_channels, self.dropout, self.use_se_block)
        else:
            model = SmileNet(self.in_channels, self.dropout)
        
        # Adjust number of classes if needed
        if self.num_classes != 2:
            # Tìm layer Linear cuối cùng trong classifier
            if hasattr(model, 'classifier'):
                for i in range(len(model.classifier) - 1, -1, -1):
                    if isinstance(model.classifier[i], nn.Linear):
                        in_features = model.classifier[i].in_features
                        model.classifier[i] = nn.Linear(in_features, self.num_classes)
                        break
        return model


def build_model(config: SmileNetConfig | None = None) -> nn.Module:
    return (config or SmileNetConfig()).build()


def load_weights(weights_path: str, device: torch.device | str = "cpu", model_name: str = "SmileNet") -> nn.Module:
    config = SmileNetConfig(model_name=model_name)
    model = build_model(config)
    state = torch.load(weights_path, map_location=device, weights_only=True)
    
    # Nếu checkpoint có key 'model_state_dict', extract nó
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"⚠️  Warning: Checkpoint không khớp hoàn toàn")
        if missing:
            print(f"  - Thiếu keys: {missing[:5]}...")  # Chỉ hiện 5 keys đầu
        if unexpected:
            print(f"  - Thừa keys: {unexpected[:5]}...")
    return model.to(device)


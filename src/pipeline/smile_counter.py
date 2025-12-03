"""Pipeline phát hiện khuôn mặt và phân loại nụ cười."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.classifier.smile_model import IMAGE_SIZE, build_model
from src.detection import YOLOFaceConfig, YOLOFaceDetector


@dataclass(slots=True)
class SmileCounterConfig:
    """Tùy chọn cấu hình."""

    face_model: str | None = "models/yolov8n-face.pt"
    classifier_weights: str = "models/smile_cnn_best.pth"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    smile_threshold: float = 0.45  # Ngưỡng 40% để coi là cười
    expand_ratio: float = 1.15


class SmileCounter:
    """Ghép YOLO + CNN để đếm nụ cười."""

    def __init__(self, config: SmileCounterConfig | None = None) -> None:
        self.config = config or SmileCounterConfig()
        face_cfg = YOLOFaceConfig(model_path=self.config.face_model, device=self.config.device)
        self.face_detector = YOLOFaceDetector(face_cfg)
        self.device = torch.device(self.config.device)
        self.model = self._load_classifier(self.config.classifier_weights)
        self.transform = transforms.Compose(
            [
                transforms.Resize(IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def _load_classifier(self, weights: str | Path) -> torch.nn.Module:
        """Load classifier model, tự động detect architecture từ checkpoint."""
        path = Path(weights)
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy trọng số tại {path}")
        
        # Thử load state dict để detect architecture
        state = torch.load(path, map_location=self.device, weights_only=True)
        
        # Nếu là dict với 'model_state_dict', extract nó
        if isinstance(state, dict) and 'model_state_dict' in state:
            state = state['model_state_dict']
        
        # Detect architecture dựa vào keys trong state dict
        # SmileNetV2 có 'layer1', 'layer2', 'layer3'
        # SmileNet chỉ có 'features'
        from src.classifier.smile_model import SmileNetConfig
        
        if any('layer1' in k for k in state.keys()):
            # SmileNetV2
            config = SmileNetConfig(model_name="SmileNetV2", use_se_block=True)
            model = config.build()
            print("🔍 Detected SmileNetV2 architecture")
        else:
            # SmileNet baseline
            config = SmileNetConfig(model_name="SmileNet")
            model = config.build()
            print("🔍 Detected SmileNet architecture")
        
        # Load weights
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"⚠️  Warning: Checkpoint có thể không khớp hoàn toàn")
        
        model.eval()
        return model.to(self.device)

    @staticmethod
    def _expand_boxes(boxes: List[Tuple[int, int, int, int, float]], shape: Tuple[int, int, int], ratio: float) -> List[Tuple[int, int, int, int, float]]:
        h, w = shape[:2]
        out: List[Tuple[int, int, int, int, float]] = []
        for x, y, bw, bh, conf in boxes:
            cx = x + bw / 2
            cy = y + bh / 2
            nw = bw * ratio
            nh = bh * ratio
            nx = max(0, int(cx - nw / 2))
            ny = max(0, int(cy - nh / 2))
            ex = min(w, int(nx + nw))
            ey = min(h, int(ny + nh))
            out.append((nx, ny, ex - nx, ey - ny, conf))
        return out

    def detect_faces(self, image_bgr: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        raw_boxes = self.face_detector.detect(image_bgr)
        return self._expand_boxes(raw_boxes, image_bgr.shape, self.config.expand_ratio)

    def _prepare_crop(self, crop_bgr: np.ndarray) -> torch.Tensor:
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(Image.fromarray(crop_rgb))
        return tensor.unsqueeze(0).to(self.device)

    def classify_crop(self, crop_bgr: np.ndarray) -> Tuple[int, float]:
        if crop_bgr.size == 0:
            return 0, 0.0
        tensor = self._prepare_crop(crop_bgr)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)
            smile_prob = float(probs[0, 1].cpu().item())
        label = 1 if smile_prob >= self.config.smile_threshold else 0
        return label, smile_prob

    def analyze_array(self, image_bgr: np.ndarray) -> Dict[str, object]:
        detections = []
        boxes = self.detect_faces(image_bgr)
        smile_count = 0
        for x, y, w, h, score in boxes:
            crop = image_bgr[y : y + h, x : x + w]
            label, prob = self.classify_crop(crop)
            smile_count += label
            detections.append(
                {
                    "box": [int(x), int(y), int(w), int(h)],
                    "confidence": float(score),
                    "is_smiling": bool(label),
                    "smile_probability": float(prob),
                }
            )
        return {
            "total_faces": len(boxes),
            "smiling_faces": smile_count,
            "detections": detections,
        }

    def annotate(self, image_bgr: np.ndarray, detections: List[Dict[str, object]]) -> np.ndarray:
        canvas = image_bgr.copy()
        for item in detections:
            x, y, w, h = item["box"]
            is_smiling = item["is_smiling"]
            prob = item["smile_probability"]
            color = (0, 200, 0) if is_smiling else (0, 0, 200)
            cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)
            text = f"Cuoi {prob:.2f}" if is_smiling else f"Khong {prob:.2f}"
            cv2.putText(canvas, text, (x, max(0, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        return canvas

    def analyze_file(self, image_path: str | Path) -> Dict[str, object]:
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Không đọc được ảnh: {image_path}")
        return self.analyze_array(image)

    def analyze_bytes(self, data: bytes) -> Dict[str, object]:
        arr = np.frombuffer(data, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Dữ liệu không phải ảnh hợp lệ")
        return self.analyze_array(image)

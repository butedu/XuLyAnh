"""Ensemble face detector combining Haar Cascade, DNN, and MTCNN for robust detection."""
import os
import sys
from typing import List, Tuple
from pathlib import Path

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.models.face_detector import BoNhanDienKhuonMat, CauHinhBoNhanDienKhuonMat

# Try to import MTCNN (optional)
try:
    from .mtcnn_detector import MTCNNFaceDetector
    HAS_MTCNN = True
except ImportError:
    HAS_MTCNN = False
 
# Try to import YOLOv8 face detector (optional)
try:
    from .yolo_face_detector import YOLOFaceDetector
    HAS_YOLO = True
except Exception:
    HAS_YOLO = False


class EnsembleFaceDetector:
    """Combine multiple face detectors to catch faces from various angles and scales.
    
    Uses:
    - Haar Cascade (fast, good for frontal faces)
    - DNN/SSD (better for angled faces, if available)
    - MTCNN (best for challenging angles, if facenet-pytorch installed)
    
    All detections are merged and deduplicated using IoU threshold.
    """

    def __init__(self, min_face_size: int = 20, iou_threshold: float = 0.2):
        """Initialize ensemble detector with multiple backends.
        
        Args:
            min_face_size: Minimum face size to detect (pixels)
            iou_threshold: IoU threshold for deduplication (0-1). Lower = more aggressive merge.
        """
        self.min_face_size = min_face_size
        self.iou_threshold = iou_threshold
        
        # Initialize Haar Cascade (always available)
        try:
            self.haar_detector = BoNhanDienKhuonMat(
                CauHinhBoNhanDienKhuonMat(backend='haar', min_confidence=0.5)
            )
        except Exception as e:
            print(f"[EnsembleFaceDetector] Warning: Haar detector failed: {e}")
            self.haar_detector = None
        
        # Initialize DNN (optional, will skip if files missing)
        self.dnn_detector = None
        try:
            self.dnn_detector = BoNhanDienKhuonMat(
                CauHinhBoNhanDienKhuonMat(backend='dnn', min_confidence=0.4)
            )
        except FileNotFoundError:
            print("[EnsembleFaceDetector] DNN detector files not found; skipping DNN backend")
        except Exception as e:
            print(f"[EnsembleFaceDetector] Warning: DNN detector failed: {e}")
        
        # Initialize MTCNN (optional)
        self.mtcnn_detector = None
        if HAS_MTCNN:
            try:
                device = 'cuda' if np.zeros(1).dtype == np.float32 else 'cpu'  # Rough check
                import torch
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                self.mtcnn_detector = MTCNNFaceDetector(device=device, min_face_size=min_face_size)
                print(f"[EnsembleFaceDetector] MTCNN detector initialized on device: {device}")
            except Exception as e:
                print(f"[EnsembleFaceDetector] Warning: MTCNN detector failed: {e}")
        else:
            print("[EnsembleFaceDetector] facenet-pytorch not installed; MTCNN skipped")

        # Initialize YOLO face detector (optional)
        self.yolo_detector = None
        if HAS_YOLO:
            try:
                # Let YOLO wrapper handle model path fallback
                self.yolo_detector = YOLOFaceDetector()
                print("[EnsembleFaceDetector] YOLOv8-face detector initialized")
            except Exception as e:
                print(f"[EnsembleFaceDetector] Warning: YOLO detector failed: {e}")
                self.yolo_detector = None

    def _convert_mtcnn_to_standard(self, mtcnn_boxes: List[Tuple]) -> List[Tuple[int, int, int, int, float]]:
        """Convert MTCNN box format (x1, y1, x2, y2) to standard (x, y, w, h, confidence)."""
        result = []
        for (x1, y1, x2, y2), conf in mtcnn_boxes:
            w = x2 - x1
            h = y2 - y1
            if w > 0 and h > 0:
                result.append((x1, y1, w, h, float(conf)))
        return result

    def _iou(self, box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
        """Calculate Intersection over Union (IoU) between two boxes (x, y, w, h format)."""
        x1_1, y1_1, w1, h1 = box1
        x2_1, y2_1 = x1_1 + w1, y1_1 + h1
        
        x1_2, y1_2, w2, h2 = box2
        x2_2, y2_2 = x1_2 + w2, y1_2 + h2
        
        # Intersection
        xi1, yi1 = max(x1_1, x1_2), max(y1_1, y1_2)
        xi2, yi2 = min(x2_1, x2_2), min(y2_1, y2_2)
        
        if xi2 < xi1 or yi2 < yi1:
            return 0.0
        
        intersection = (xi2 - xi1) * (yi2 - yi1)
        
        # Union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0

    def _deduplicate_boxes(self, all_boxes: List[Tuple[int, int, int, int, float]]) -> List[Tuple[int, int, int, int, float]]:
        """Remove duplicate detections.

        Prefer using OpenCV NMSBoxes (faster and robust). If NMS is not available
        or fails, falls back to greedy IoU-based deduplication using
        `self.iou_threshold`.
        """
        if not all_boxes:
            return []

        try:
            # Prepare boxes and scores for NMS
            boxes_xywh = [[int(b[0]), int(b[1]), int(b[2]), int(b[3])] for b in all_boxes]
            scores = [float(b[4]) for b in all_boxes]
            # OpenCV expects scoreThreshold and nmsThreshold; we'll use 0.2 and 0.3 defaults
            indices = cv2.dnn.NMSBoxes(boxes_xywh, scores, score_threshold=0.2, nms_threshold=0.3)
            kept = []
            if len(indices) > 0:
                # indices can be list of lists or flat array
                if isinstance(indices, (list, tuple)):
                    flat = [int(i[0]) if isinstance(i, (list, tuple)) else int(i) for i in indices]
                else:
                    flat = indices.flatten().tolist()
                for i in flat:
                    kept.append(all_boxes[i])
            else:
                # nothing passed NMS, fall back to greedy
                raise RuntimeError('NMS returned no indices')
            return kept
        except Exception:
            # Fallback: greedy IoU-based dedup
            sorted_boxes = sorted(all_boxes, key=lambda b: b[4], reverse=True)
            kept = []
            for current in sorted_boxes:
                is_duplicate = False
                for kept_box in kept:
                    iou_val = self._iou(current[:4], kept_box[:4])
                    if iou_val > self.iou_threshold:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    kept.append(current)
            return kept

    def detect(self, image_bgr: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """Detect faces using ensemble of multiple detectors.
        
        Args:
            image_bgr: Image in BGR format (OpenCV)
            
        Returns:
            List of detections (x, y, w, h, confidence), deduplicated and sorted by confidence.
        """
        all_detections = []
        
        # Haar Cascade
        if self.haar_detector:
            try:
                haar_boxes = self.haar_detector.phat_hien(image_bgr)
                all_detections.extend(haar_boxes)
                print(f"[EnsembleFaceDetector] Haar detected {len(haar_boxes)} faces")
            except Exception as e:
                print(f"[EnsembleFaceDetector] Haar detection error: {e}")
        
        # DNN
        if self.dnn_detector:
            try:
                dnn_boxes = self.dnn_detector.phat_hien(image_bgr)
                all_detections.extend(dnn_boxes)
                print(f"[EnsembleFaceDetector] DNN detected {len(dnn_boxes)} faces")
            except Exception as e:
                print(f"[EnsembleFaceDetector] DNN detection error: {e}")
        
        # MTCNN
        if self.mtcnn_detector:
            try:
                from PIL import Image
                pil_image = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
                mtcnn_boxes = self.mtcnn_detector.detect(pil_image)
                mtcnn_standard = self._convert_mtcnn_to_standard(mtcnn_boxes)
                all_detections.extend(mtcnn_standard)
                print(f"[EnsembleFaceDetector] MTCNN detected {len(mtcnn_standard)} faces")
            except Exception as e:
                print(f"[EnsembleFaceDetector] MTCNN detection error: {e}")
        
        # YOLOv8-face
        if self.yolo_detector:
            try:
                yolo_boxes = self.yolo_detector.detect_bgr(image_bgr)
                all_detections.extend(yolo_boxes)
                print(f"[EnsembleFaceDetector] YOLO detected {len(yolo_boxes)} faces")
            except Exception as e:
                print(f"[EnsembleFaceDetector] YOLO detection error: {e}")
        
        print(f"[EnsembleFaceDetector] Total detections before dedup: {len(all_detections)}")
        
        # Deduplicate
        final_detections = self._deduplicate_boxes(all_detections)
        
        # Sort by confidence descending
        final_detections = sorted(final_detections, key=lambda b: b[4], reverse=True)
        
        print(f"[EnsembleFaceDetector] Final detections after dedup: {len(final_detections)}")
        
        return final_detections

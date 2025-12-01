"""YOLOv8-face detector wrapper using Ultralytics YOLO API.

This module provides a simple wrapper `YOLOFaceDetector` that attempts to load
`ultralytics` and a YOLO model name (default `yolov8n-face.pt` if present).

It is written to fail gracefully when `ultralytics` or the model file is not
available: the ensemble will fall back to other detectors.
"""
from __future__ import annotations
import os
import sys
from typing import List, Tuple

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


class YOLOFaceDetector:
    """Wrapper around Ultralytics YOLO models for face detection.

    Usage:
        det = YOLOFaceDetector(model_path=None)
        boxes = det.detect_bgr(image_bgr)

    Returns:
        list of (x, y, w, h, confidence)
    """

    def __init__(self, model_path: str | None = None, device: str = 'cpu') -> None:
        if YOLO is None:
            raise ImportError('ultralytics is required for YOLOFaceDetector. Install via `pip install ultralytics`')
        # If model_path is omitted, try to use a built-in name (user should provide path)
        self.model_path = model_path
        if model_path is None:
            # common name for a small face-specific model; user can replace
            self.model_path = 'yolov8n-face.pt'
        # initialize model; let YOLO raise if model file not found
        try:
            self.model = YOLO(self.model_path)
            # set device
            self.model.fuse()
            # If device arg is supported, user can re-initialize; keep basic
        except Exception as e:
            raise RuntimeError(f'Failed to load YOLO model at {self.model_path}: {e}')

    def detect_bgr(self, image_bgr) -> List[Tuple[int, int, int, int, float]]:
        """Run inference on an OpenCV BGR image.

        Returns boxes in (x, y, w, h, confidence)
        """
        # YOLO API accepts RGB images or paths; convert to RGB
        import numpy as np
        img_rgb = np.ascontiguousarray(image_bgr[..., ::-1])
        # Inference
        results = self.model(img_rgb, imgsz=640, conf=0.25, iou=0.45, device=self.model.model.device)
        # results may be a list; take first
        boxes_out = []
        try:
            r = results[0]
            # r.boxes.xyxy, r.boxes.conf
            if hasattr(r, 'boxes') and len(r.boxes) > 0:
                xyxy = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                for (x1, y1, x2, y2), conf in zip(xyxy, confs):
                    x = int(x1)
                    y = int(y1)
                    w = int(max(0, x2 - x1))
                    h = int(max(0, y2 - y1))
                    boxes_out.append((x, y, w, h, float(conf)))
        except Exception:
            # In case API differs, try attribute-based access
            try:
                for det in results:
                    for box in det.boxes:
                        x1, y1, x2, y2 = box.xyxy[0]
                        conf = float(box.conf[0])
                        x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                        boxes_out.append((x, y, w, h, conf))
            except Exception:
                pass
        return boxes_out

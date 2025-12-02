"""Giữ tên cũ cho YOLOFaceDetector, chuyển sang module detection mới."""

from src.detection.yolo import YOLOFaceConfig, YOLOFaceDetector  # noqa: F401

__all__ = ["YOLOFaceDetector", "YOLOFaceConfig"]

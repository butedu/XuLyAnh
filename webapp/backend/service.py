"""Dịch vụ FastAPI sử dụng pipeline mới."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

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

    def _ve_tong_quan(self, frame: np.ndarray, total: int, smiles: int) -> None:
        cv2.rectangle(frame, (12, 12), (280, 82), (16, 24, 36), -1)
        cv2.putText(frame, "SmileCounter", (24, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2)
        cv2.putText(
            frame,
            f"Faces: {total} | Smiling: {smiles}",
            (24, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (204, 221, 255),
            2,
        )

    def xu_ly_video_file(
        self,
        duong_dan_vao: str | Path,
        duong_dan_ra: str | Path,
        frame_skip: int = 0,
        resize: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, object]:
        cap = cv2.VideoCapture(str(duong_dan_vao))
        if not cap.isOpened():
            raise ValueError("Không thể mở video đầu vào")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if resize is not None:
            width, height = resize

        if width <= 0 or height <= 0:
            cap.release()
            raise ValueError("Kích thước video không hợp lệ")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(duong_dan_ra), fourcc, fps, (width, height))
        if not writer.isOpened():
            cap.release()
            raise ValueError("Không thể tạo video đầu ra")

        stats = {
            "total_frames": 0,
            "processed_frames": 0,
            "faces_detected": 0,
            "smiles_detected": 0,
            "frame_skip": max(frame_skip, 0),
            "fps": round(float(fps), 2),
            "resize": list(resize) if resize else None,
        }

        frame_index = 0

        try:
            while True:
                grabbed, frame = cap.read()
                if not grabbed:
                    break
                stats["total_frames"] += 1

                if resize is not None:
                    frame = cv2.resize(frame, resize)

                if stats["frame_skip"] and frame_index % (stats["frame_skip"] + 1) != 0:
                    annotated = frame
                else:
                    summary = self.pipeline.analyze_array(frame)
                    annotated = self.pipeline.annotate(frame, summary.get("detections", []))
                    self._ve_tong_quan(annotated, summary["total_faces"], summary["smiling_faces"])
                    stats["processed_frames"] += 1
                    stats["faces_detected"] += summary["total_faces"]
                    stats["smiles_detected"] += summary["smiling_faces"]

                writer.write(annotated)
                frame_index += 1
        finally:
            cap.release()
            writer.release()

        if stats["processed_frames"]:
            stats["avg_faces_per_processed_frame"] = round(
                stats["faces_detected"] / stats["processed_frames"],
                3,
            )
            stats["avg_smiles_per_processed_frame"] = round(
                stats["smiles_detected"] / stats["processed_frames"],
                3,
            )
        else:
            stats["avg_faces_per_processed_frame"] = 0.0
            stats["avg_smiles_per_processed_frame"] = 0.0

        if stats["fps"] > 0:
            stats["duration_seconds"] = round(stats["total_frames"] / stats["fps"], 2)
        else:
            stats["duration_seconds"] = None

        return stats


# Alias giữ tương thích
SmileService = DichVuNhanDienCuoi
analyze_image_bytes = DichVuNhanDienCuoi.phan_tich_anh_bytes
annotate_image = DichVuNhanDienCuoi.chu_thich_anh

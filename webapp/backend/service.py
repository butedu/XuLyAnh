"""Dịch vụ FastAPI sử dụng pipeline mới."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

# Pipeline chính để detect mặt + phân loại cười
from src.pipeline.smile_counter import SmileCounter, SmileCounterConfig
# (Hiện chưa dùng trực tiếp, nhưng để sẵn cho tracking nâng cao)
from src.tracking.face_tracker import SimpleFaceTracker


class DichVuNhanDienCuoi:
    """Bao lớp SmileCounter cho web API."""

    def __init__(
        self,
        duong_dan_mo_hinh: str | Path | None = None,
        duong_dan_face: str | Path | None = "models/yolov8n-face.pt",
        device: str | None = None,
    ) -> None:
        # Tạo config cho SmileCounter
        config = SmileCounterConfig(
            # Trọng số model phân loại cười
            classifier_weights=str(duong_dan_mo_hinh or "models/smile_cnn_best.pth"),
            # Model detect face (YOLOv8-face)
            face_model=str(duong_dan_face) if duong_dan_face is not None else None,
            # Tự chọn device: CUDA nếu có, không thì CPU
            device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
        )
        # Khởi tạo pipeline xử lý chính
        self.pipeline = SmileCounter(config)

    def phan_tich_anh_bytes(self, anh_bytes: bytes) -> Dict[str, object]:
        """
        Nhận ảnh dưới dạng bytes (upload từ API),
        decode sang OpenCV image và chạy pipeline phân tích
        """
        # Chuyển bytes → numpy array
        arr = np.frombuffer(anh_bytes, dtype=np.uint8)
        # Decode ảnh (BGR)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Tệp tải lên không phải ảnh hợp lệ")

        # Phân tích ảnh bằng pipeline
        return self.pipeline.analyze_array(image)

    def chu_thich_anh(self, anh_bytes: bytes, ket_qua: Dict[str, object]) -> bytes:
        """
        Vẽ bounding box + nhãn cười lên ảnh
        và trả về ảnh đã annotate dưới dạng bytes
        """
        arr = np.frombuffer(anh_bytes, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Tệp tải lên không phải ảnh hợp lệ")

        # Vẽ annotation dựa trên kết quả detect
        annotated = self.pipeline.annotate(image, ket_qua.get("detections", []))

        # Encode ảnh sang JPEG
        ok, buffer = cv2.imencode(".jpg", annotated)
        if not ok:
            raise ValueError("Không thể mã hóa ảnh")
        return bytes(buffer)

    def _ve_tong_quan(self, frame: np.ndarray, total: int, smiles: int) -> None:
        """
        Vẽ overlay thông tin tổng quan (faces, smiles) lên frame
        """
        cv2.rectangle(frame, (12, 12), (280, 82), (16, 24, 36), -1)
        cv2.putText(frame, "SmileCounter", (24, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2)
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
        """
        Xử lý video:
        - Detect mặt + cười từng frame
        - Ghi video output đã annotate
        - Trả về thống kê tổng hợp
        """
        cap = cv2.VideoCapture(str(duong_dan_vao))
        if not cap.isOpened():
            raise ValueError("Không thể mở video đầu vào")

        # Lấy thông tin video
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Resize nếu được chỉ định
        if resize is not None:
            width, height = resize

        if width <= 0 or height <= 0:
            cap.release()
            raise ValueError("Kích thước video không hợp lệ")

        # Codec H.264 (avc1) để xem trên trình duyệt
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(str(duong_dan_ra), fourcc, fps, (width, height))
        if not writer.isOpened():
            cap.release()
            raise ValueError("Không thể tạo video đầu ra")

        # Thống kê
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

                # Skip frame để tăng tốc
                if stats["frame_skip"] and frame_index % (stats["frame_skip"] + 1) != 0:
                    annotated = frame
                else:
                    # Phân tích frame
                    summary = self.pipeline.analyze_array(frame)
                    annotated = self.pipeline.annotate(frame, summary.get("detections", []))
                    self._ve_tong_quan(
                        annotated,
                        summary["total_faces"],
                        summary["smiling_faces"],
                    )
                    stats["processed_frames"] += 1
                    stats["faces_detected"] += summary["total_faces"]
                    stats["smiles_detected"] += summary["smiling_faces"]

                writer.write(annotated)
                frame_index += 1
        finally:
            cap.release()
            writer.release()

        # Tính trung bình
        if stats["processed_frames"]:
            stats["avg_faces_per_processed_frame"] = round(
                stats["faces_detected"] / stats["processed_frames"], 3
            )
            stats["avg_smiles_per_processed_frame"] = round(
                stats["smiles_detected"] / stats["processed_frames"], 3
            )
        else:
            stats["avg_faces_per_processed_frame"] = 0.0
            stats["avg_smiles_per_processed_frame"] = 0.0

        # Thời lượng video
        if stats["fps"] > 0:
            stats["duration_seconds"] = round(stats["total_frames"] / stats["fps"], 2)
        else:
            stats["duration_seconds"] = None

        return stats

    def xu_ly_video_tracking(
        self,
        duong_dan_vao: str | Path,
        duong_dan_ra: str | Path,
        process_every: int = 3,
        frame_skip: int = 0,
        scene_threshold: float = 0.35,
        resize: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, object]:
        """
        Xử lý video với tracking đơn giản:
        - Mỗi face detection được gán ID
        - Theo dõi smile theo thời gian
        - Lưu snapshot nụ cười đẹp nhất
        """
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

        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(str(duong_dan_ra), fourcc, fps, (width, height))
        if not writer.isOpened():
            cap.release()
            raise ValueError("Không thể tạo video đầu ra")

        # Track đơn giản: mỗi ID lưu lịch sử bbox + thống kê
        tracks = {}
        next_id = 1

        # Folder lưu ảnh snapshot nụ cười
        snapshot_dir = Path(duong_dan_ra).parent / f"{Path(duong_dan_ra).stem}_snapshots"
        snapshot_dir.mkdir(exist_ok=True)

        stats = {
            "total_frames": 0,
            "processed_frames": 0,
            "faces_detected": 0,
            "smiles_detected": 0,
            "process_every": process_every,
            "fps": round(float(fps), 2),
            "resize": list(resize) if resize else None,
            "people": [],
        }

        frame_index = 0

        try:
            while True:
                grabbed, frame = cap.read()
                if not grabbed:
                    break

                stats["total_frames"] += 1

                # Skip frame
                if frame_skip > 0 and (stats["total_frames"] - 1) % (frame_skip + 1) != 0:
                    continue

                if resize is not None:
                    frame = cv2.resize(frame, resize)

                annotated = frame.copy()

                # Chỉ detect mỗi process_every frame
                if frame_index % process_every == 0:
                    try:
                        summary = self.pipeline.analyze_array(frame)
                        detections = summary.get("detections", [])

                        stats["processed_frames"] += 1
                        stats["faces_detected"] += len(detections)
                        stats["smiles_detected"] += sum(1 for d in detections if d["is_smiling"])

                        # Gán ID cho từng detection
                        for det in detections:
                            x, y, w, h = det["box"]
                            smile_prob = det["smile_probability"]
                            is_smiling = det["is_smiling"]

                            matched_id = None
                            min_distance = 100

                            # Match bbox gần nhất
                            for track_id, track_data in tracks.items():
                                if track_data["bbox_history"]:
                                    lx, ly, lw, lh = track_data["bbox_history"][-1]
                                    dist = ((x + w/2) - (lx + lw/2))**2 + ((y + h/2) - (ly + lh/2))**2
                                    dist = dist ** 0.5
                                    if dist < min_distance:
                                        min_distance = dist
                                        matched_id = track_id

                            # Tạo track mới nếu không match
                            if matched_id is None:
                                matched_id = next_id
                                tracks[matched_id] = {
                                    "frames": 0,
                                    "smile_frames": 0,
                                    "bbox_history": [],
                                    "best_smile_frame": None,
                                    "best_smile_prob": 0.0,
                                }
                                next_id += 1

                            # Update track
                            tracks[matched_id]["frames"] += 1
                            if is_smiling:
                                tracks[matched_id]["smile_frames"] += 1
                                if smile_prob > tracks[matched_id]["best_smile_prob"]:
                                    tracks[matched_id]["best_smile_prob"] = smile_prob
                                    face_crop = frame[y:y+h, x:x+w]
                                    snapshot_path = snapshot_dir / f"person_{matched_id}_smile.jpg"
                                    cv2.imwrite(str(snapshot_path), face_crop)
                                    tracks[matched_id]["best_smile_frame"] = snapshot_path.name

                            tracks[matched_id]["bbox_history"].append((x, y, w, h))
                            if len(tracks[matched_id]["bbox_history"]) > 10:
                                tracks[matched_id]["bbox_history"].pop(0)

                            # Vẽ bbox + label
                            smile_ratio = tracks[matched_id]["smile_frames"] / tracks[matched_id]["frames"]
                            color = (0, 255, 0) if is_smiling else (0, 0, 255)
                            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
                            label = f"ID {matched_id} | {smile_prob*100:.0f}% | {smile_ratio*100:.0f}%"
                            cv2.putText(annotated, label, (x, y - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    except Exception as e:
                        print(f"⚠️ Error processing frame {frame_index}: {e}")
                        import traceback
                        traceback.print_exc()

                # Overlay tổng quan
                active_tracks = len([t for t in tracks.values() if t["frames"] > 0])
                smiling_tracks = len(
                    [t for t in tracks.values()
                     if t["frames"] > 0 and t["smile_frames"] / t["frames"] >= 0.3]
                )

                cv2.rectangle(annotated, (12, 12), (320, 82), (16, 24, 36), -1)
                cv2.putText(annotated, "SmileCounter + Tracking", (24, 38),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2)
                cv2.putText(
                    annotated,
                    f"People: {active_tracks} | Smiling: {smiling_tracks}",
                    (24, 68),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (204, 221, 255),
                    2,
                )

                writer.write(annotated)
                frame_index += 1

        finally:
            cap.release()
            writer.release()

        # Tổng hợp thống kê theo từng người
        people_stats = []
        for track_id, track_data in tracks.items():
            if track_data["frames"] < 3:
                continue

            smile_ratio = track_data["smile_frames"] / track_data["frames"]
            people_stats.append({
                "id": track_id,
                "total_frames": track_data["frames"],
                "smile_frames": track_data["smile_frames"],
                "smile_ratio": round(smile_ratio, 3),
                "duration": round(track_data["frames"] / fps, 2),
                "smile_duration": round(track_data["smile_frames"] / fps, 2),
                "is_smiling": smile_ratio >= 0.3,
                "snapshot": track_data.get("best_smile_frame"),
            })

        stats["people"] = people_stats
        stats["total_people"] = len(people_stats)
        stats["people_smiling"] = sum(1 for p in people_stats if p["is_smiling"])
        stats["snapshot_dir"] = str(snapshot_dir.name)

        if stats["processed_frames"] > 0:
            stats["avg_faces_per_processed_frame"] = round(stats["faces_detected"] / stats["processed_frames"], 2)
            stats["avg_smiles_per_processed_frame"] = round(stats["smiles_detected"] / stats["processed_frames"], 2)

        if stats["fps"] > 0:
            stats["duration_seconds"] = round(stats["total_frames"] / stats["fps"], 2)
        else:
            stats["duration_seconds"] = None

        return stats


# Alias giữ tương thích ngược với code cũ
SmileService = DichVuNhanDienCuoi
analyze_image_bytes = DichVuNhanDienCuoi.phan_tich_anh_bytes
annotate_image = DichVuNhanDienCuoi.chu_thich_anh

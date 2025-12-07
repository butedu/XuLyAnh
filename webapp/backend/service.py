"""Dịch vụ FastAPI sử dụng pipeline mới."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from src.pipeline.smile_counter import SmileCounter, SmileCounterConfig
from src.tracking.face_tracker import SimpleFaceTracker


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

        # Sử dụng H.264 codec để tương thích với trình duyệt
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
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

    def xu_ly_video_tracking(
        self,
        duong_dan_vao: str | Path,
        duong_dan_ra: str | Path,
        process_every: int = 3,
        frame_skip: int = 0,
        scene_threshold: float = 0.35,
        resize: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, object]:
        """Xử lý video với face tracking đơn giản - mỗi face detection tạo 1 track riêng."""
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

        # Sử dụng H.264 codec để tương thích với trình duyệt
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(str(duong_dan_ra), fourcc, fps, (width, height))
        if not writer.isOpened():
            cap.release()
            raise ValueError("Không thể tạo video đầu ra")

        # Simple tracking: dictionary của {id: {frames, smile_frames, bbox_history}}
        tracks = {}
        next_id = 1
        
        # Tạo folder để lưu snapshots
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
                
                # Skip frames if frame_skip > 0
                if frame_skip > 0 and (stats["total_frames"] - 1) % (frame_skip + 1) != 0:
                    continue

                if resize is not None:
                    frame = cv2.resize(frame, resize)

                # Detection mỗi process_every frames
                annotated = frame.copy()
                if frame_index % process_every == 0:
                    try:
                        summary = self.pipeline.analyze_array(frame)
                        detections = summary.get("detections", [])
                        stats["processed_frames"] += 1
                        stats["faces_detected"] += len(detections)
                        stats["smiles_detected"] += sum(1 for d in detections if d["is_smiling"])
                        
                        # Simple tracking: assign ID to each detection
                        for det in detections:
                            x, y, w, h = det["box"]
                            smile_prob = det["smile_probability"]
                            is_smiling = det["is_smiling"]
                            
                            # Tìm track gần nhất hoặc tạo mới
                            matched_id = None
                            min_distance = 100  # threshold
                            
                            for track_id, track_data in tracks.items():
                                if len(track_data["bbox_history"]) > 0:
                                    last_bbox = track_data["bbox_history"][-1]
                                    lx, ly, lw, lh = last_bbox
                                    # Distance giữa 2 center
                                    dist = ((x + w/2) - (lx + lw/2))**2 + ((y + h/2) - (ly + lh/2))**2
                                    dist = dist ** 0.5
                                    if dist < min_distance:
                                        min_distance = dist
                                        matched_id = track_id
                            
                            # Nếu không match được, tạo track mới
                            if matched_id is None:
                                matched_id = next_id
                                tracks[matched_id] = {
                                    "frames": 0,
                                    "smile_frames": 0,
                                    "bbox_history": [],
                                    "best_smile_frame": None,
                                    "best_smile_prob": 0.0
                                }
                                next_id += 1
                            
                            # Update track
                            tracks[matched_id]["frames"] += 1
                            if is_smiling:
                                tracks[matched_id]["smile_frames"] += 1
                                # Lưu frame có smile_prob cao nhất
                                if smile_prob > tracks[matched_id]["best_smile_prob"]:
                                    tracks[matched_id]["best_smile_prob"] = smile_prob
                                    # Crop và lưu ảnh
                                    face_crop = frame[y:y+h, x:x+w]
                                    snapshot_path = snapshot_dir / f"person_{matched_id}_smile.jpg"
                                    cv2.imwrite(str(snapshot_path), face_crop)
                                    tracks[matched_id]["best_smile_frame"] = str(snapshot_path.name)
                            tracks[matched_id]["bbox_history"].append((x, y, w, h))
                            # Giữ chỉ 10 bbox gần nhất
                            if len(tracks[matched_id]["bbox_history"]) > 10:
                                tracks[matched_id]["bbox_history"].pop(0)
                            
                            # Vẽ lên frame
                            smile_ratio = tracks[matched_id]["smile_frames"] / tracks[matched_id]["frames"]
                            color = (0, 255, 0) if is_smiling else (0, 0, 255)
                            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
                            label = f"ID {matched_id} | {smile_prob*100:.0f}% | {smile_ratio*100:.0f}%"
                            cv2.putText(annotated, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    except Exception as e:
                        print(f"⚠️ Error processing frame {frame_index}: {e}")
                        import traceback
                        traceback.print_exc()

                # Vẽ tổng quan
                active_tracks = len([t for t in tracks.values() if t["frames"] > 0])
                smiling_tracks = len([t for t in tracks.values() if t["frames"] > 0 and t["smile_frames"] / t["frames"] >= 0.3])
                cv2.rectangle(annotated, (12, 12), (320, 82), (16, 24, 36), -1)
                cv2.putText(annotated, "SmileCounter + Tracking", (24, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2)
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

        # Thu thập thống kê
        print(f"📊 Total tracks collected: {len(tracks)}")
        people_stats = []
        for track_id, track_data in tracks.items():
            if track_data["frames"] < 3:  # Bỏ qua tracks quá ngắn
                continue
            
            smile_ratio = track_data["smile_frames"] / track_data["frames"]
            duration = track_data["frames"] / fps
            smile_duration = track_data["smile_frames"] / fps
            
            people_stats.append({
                "id": track_id,
                "total_frames": track_data["frames"],
                "smile_frames": track_data["smile_frames"],
                "smile_ratio": round(smile_ratio, 3),
                "duration": round(duration, 2),
                "smile_duration": round(smile_duration, 2),
                "is_smiling": smile_ratio >= 0.3,
                "snapshot": track_data.get("best_smile_frame"),
            })
            print(f"  Track {track_id}: frames={track_data['frames']}, smile_ratio={smile_ratio:.2%}")
        
        print(f"📈 Final people_stats count: {len(people_stats)}")
        
        stats["people"] = people_stats
        stats["total_people"] = len(people_stats)
        stats["people_smiling"] = sum(1 for p in people_stats if p["is_smiling"])
        stats["snapshot_dir"] = str(snapshot_dir.name)
        
        # Tính avg per processed frame
        if stats["processed_frames"] > 0:
            stats["avg_faces_per_processed_frame"] = round(stats["faces_detected"] / stats["processed_frames"], 2)
            stats["avg_smiles_per_processed_frame"] = round(stats["smiles_detected"] / stats["processed_frames"], 2)

        if stats["fps"] > 0:
            stats["duration_seconds"] = round(stats["total_frames"] / stats["fps"], 2)
        else:
            stats["duration_seconds"] = None

        return stats

        if stats["fps"] > 0:
            stats["duration_seconds"] = round(stats["total_frames"] / stats["fps"], 2)
        else:
            stats["duration_seconds"] = None

        return stats


# Alias giữ tương thích
SmileService = DichVuNhanDienCuoi
analyze_image_bytes = DichVuNhanDienCuoi.phan_tich_anh_bytes
annotate_image = DichVuNhanDienCuoi.chu_thich_anh

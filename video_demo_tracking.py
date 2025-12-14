"""
Video processing với face tracking để thống kê chính xác.
Sử dụng: python video_demo_tracking.py input.mp4 --output output.mp4
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from tqdm import tqdm

# Pipeline xử lý: YOLO-face + SmileNet
from src.pipeline.smile_counter import SmileCounter, SmileCounterConfig

# Bộ tracker khuôn mặt đơn giản
from src.tracking.face_tracker import SimpleFaceTracker, detect_scene_change


def parse_args() -> argparse.Namespace:
    """Parse các tham số dòng lệnh cho chương trình xử lý video."""
    parser = argparse.ArgumentParser(description="Xử lý video với face tracking")

    # Đường dẫn video input
    parser.add_argument("video", type=Path, help="Đường dẫn video đầu vào")

    # File video output
    parser.add_argument("--output", type=Path, default=Path("output_tracked.mp4"),
                        help="Video đầu ra")

    # Trọng số classifier SmileNet
    parser.add_argument("--weights", type=Path, default=Path("models/smile_cnn_best.pth"))

    # Model YOLO-face
    parser.add_argument("--face-model", type=str, default="models/yolov8n-face.pt")

    # Device chạy model
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Chỉ xử lý mỗi N frames để giảm tải
    parser.add_argument("--process-every", type=int, default=3, help="Xử lý mỗi N frames (giảm tải)")

    # Ngưỡng phát hiện chuyển cảnh
    parser.add_argument("--scene-threshold", type=float, default=0.35,
                        help="Ngưỡng phát hiện scene change")

    return parser.parse_args()


def draw_tracked_face(
    frame: np.ndarray,
    track_id: int,
    bbox: tuple,
    smile_prob: float,
    is_smiling: bool,
    smile_ratio: float
) -> None:
    """Vẽ bounding box, ID và thông tin cười cho từng face tracking."""
    x, y, w, h = bbox

    # Xanh nếu cười, đỏ nếu không
    color = (0, 255, 0) if is_smiling else (0, 0, 255)

    # Vẽ bounding box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # Vùng nền cho text
    label_bg_y = max(y - 35, 10)
    cv2.rectangle(frame, (x, label_bg_y), (x + w, y), color, -1)

    # ID + xác suất cười
    label = f"ID:{track_id} P:{smile_prob:.2f}"
    cv2.putText(frame, label, (x + 5, y - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Tỷ lệ thời gian cười
    ratio_text = f"Smile: {smile_ratio:.1%}"
    cv2.putText(frame, ratio_text, (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


def draw_statistics(
    frame: np.ndarray,
    stats: dict,
    fps: float,
    frame_idx: int,
    total_frames: int
) -> None:
    """Vẽ thống kê tổng quan (FPS, số người, số người cười, v.v.)."""
    h, w = frame.shape[:2]

    # Background box
    cv2.rectangle(frame, (10, 10), (400, 180), (16, 24, 36), -1)

    # Header
    cv2.putText(frame, "Video Smile Tracking", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    y_offset = 65

    # Thông tin frame hiện tại
    cv2.putText(frame, f"Frame: {frame_idx}/{total_frames}", (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    y_offset += 25

    # FPS trung bình
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    y_offset += 25

    # Tổng số người từng xuất hiện
    total_people = stats.get('total_people', 0)
    cv2.putText(frame, f"Total People: {total_people}", (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    y_offset += 30

    # Số người đang cười
    people_smiling = stats.get('people_smiling', 0)
    cv2.putText(frame, f"People Smiling: {people_smiling}", (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


def process_video(args: argparse.Namespace) -> None:
    """Luồng xử lý video: detect → classify → track → thống kê → xuất video."""
    if not args.video.exists():
        raise FileNotFoundError(f"Không tìm thấy video: {args.video}")

    print(f"🎬 Xử lý video: {args.video}")
    print(f"📊 Processing every {args.process_every} frames")
    print(f"🎯 Device: {args.device}")

    # Khởi tạo pipeline YOLO + SmileNet
    config = SmileCounterConfig(
        face_model=str(args.face_model),
        classifier_weights=str(args.weights),
        device=args.device,
    )
    counter = SmileCounter(config)

    # Khởi tạo face tracker
    tracker = SimpleFaceTracker(
        iou_threshold=0.3,
        max_age=30,            # mất tín hiệu quá 1 giây (nếu FPS=30)
        min_hits=3,            # cần 3 lần detect trùng nhau để xác nhận object
        distance_threshold=150 # ngưỡng match theo khoảng cách
    )

    # Mở video input
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise ValueError("Không thể mở video")

    # Thông tin video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"📹 Video info: {width}x{height} @ {fps} FPS, {total_frames} frames")

    # Chuẩn bị writer xuất video
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(args.output), fourcc, fps, (width, height))

    # Biến lưu trạng thái
    frame_idx = 0
    prev_frame = None
    last_detections = []
    processing_times = []

    pbar = tqdm(total=total_frames, desc="Processing")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            start_time = time.time()

            # Phát hiện chuyển cảnh → reset tracker
            if prev_frame is not None:
                is_scene_change = detect_scene_change(prev_frame, frame, args.scene_threshold)
                if is_scene_change:
                    print(f"\n🔄 Scene change detected at frame {frame_idx}, resetting tracker")
                    tracker.reset()
                    last_detections = []

            # Chỉ detect trên 1 số frame
            if frame_idx % args.process_every == 0:
                summary = counter.analyze_array(frame)
                detections = summary.get('detections', [])

                # Chuyển bbox + smile info cho tracker
                tracker_detections = [
                    {
                        'bbox': det['bbox'],
                        'smile_probability': det['smile_probability'],
                        'is_smiling': det['is_smiling']
                    }
                    for det in detections
                ]
                last_detections = tracker_detections
            else:
                tracker_detections = last_detections  # dùng kết quả trước

            # Cập nhật tracker
            confirmed_tracks = tracker.update(tracker_detections)

            # Vẽ từng face theo ID
            for track in confirmed_tracks:
                draw_tracked_face(
                    frame,
                    track.track_id,
                    track.bbox,
                    track.smile_probability,
                    track.is_smiling,
                    track.get_smile_ratio()
                )

            # Tính FPS trung bình 30 frame gần nhất
            processing_time = time.time() - start_time
            processing_times.append(processing_time)
            if len(processing_times) > 30:
                processing_times.pop(0)
            avg_fps = 1.0 / (sum(processing_times) / len(processing_times))

            # Lấy thống kê tracking
            stats = tracker.get_statistics()

            # Vẽ thống kê
            draw_statistics(frame, stats, avg_fps, frame_idx, total_frames)

            # Xuất frame
            out.write(frame)

            prev_frame = frame.copy()
            pbar.update(1)

    finally:
        cap.release()
        out.release()
        pbar.close()

    # Xuất thống kê cuối video
    print("\n" + "="*60)
    print("📊 Final Statistics")
    print("="*60)

    final_stats = tracker.get_statistics()
    print(f"Total People Tracked: {final_stats['total_people']}")
    print(f"People Smiling: {final_stats['people_smiling']}")
    print()

    # In thống kê chi tiết cho từng track
    if final_stats['tracks']:
        print("Per-person breakdown:")
        for track_info in final_stats['tracks']:
            track_id = track_info['track_id']
            total_frames = track_info['total_frames']
            smile_frames = track_info['smile_frames']
            smile_ratio = track_info['smile_ratio']
            duration = total_frames / fps
            smile_duration = smile_frames / fps

            print(f"  ID {track_id}:")
            print(f"    - Duration: {duration:.1f}s ({total_frames} frames)")
            print(f"    - Smiling: {smile_duration:.1f}s ({smile_frames} frames, {smile_ratio:.1%})")
            print(f"    - Status: {'😊 Smiling' if track_info['is_smiling'] else '😐 Neutral'}")
            print()

    print(f" Video đã được lưu: {args.output}")
    print("="*60)


def main() -> None:
    """Entry point của script."""
    args = parse_args()
    process_video(args)


if __name__ == "__main__":
    main()

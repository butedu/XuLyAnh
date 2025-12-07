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

from src.pipeline.smile_counter import SmileCounter, SmileCounterConfig
from src.tracking.face_tracker import SimpleFaceTracker, detect_scene_change


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Xử lý video với face tracking")
    parser.add_argument("video", type=Path, help="Đường dẫn video đầu vào")
    parser.add_argument("--output", type=Path, default=Path("output_tracked.mp4"), help="Video đầu ra")
    parser.add_argument("--weights", type=Path, default=Path("models/smile_cnn_best.pth"))
    parser.add_argument("--face-model", type=str, default="models/yolov8n-face.pt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--process-every", type=int, default=3, help="Xử lý mỗi N frames (giảm tải)")
    parser.add_argument("--scene-threshold", type=float, default=0.35, help="Ngưỡng phát hiện scene change")
    return parser.parse_args()


def draw_tracked_face(
    frame: np.ndarray,
    track_id: int,
    bbox: tuple,
    smile_prob: float,
    is_smiling: bool,
    smile_ratio: float
) -> None:
    """Vẽ bounding box với thông tin tracking."""
    x, y, w, h = bbox
    
    # Màu: xanh lá nếu cười, đỏ nếu không
    color = (0, 255, 0) if is_smiling else (0, 0, 255)
    
    # Vẽ bbox
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    
    # ID + smile info
    label = f"ID:{track_id} P:{smile_prob:.2f}"
    label_bg_y = max(y - 35, 10)
    cv2.rectangle(frame, (x, label_bg_y), (x + w, y), color, -1)
    cv2.putText(frame, label, (x + 5, y - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Smile ratio
    ratio_text = f"Smile: {smile_ratio:.1%}"
    cv2.putText(frame, ratio_text, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


def draw_statistics(
    frame: np.ndarray,
    stats: dict,
    fps: float,
    frame_idx: int,
    total_frames: int
) -> None:
    """Vẽ thống kê tổng quan lên frame."""
    h, w = frame.shape[:2]
    
    # Background cho stats
    cv2.rectangle(frame, (10, 10), (400, 180), (16, 24, 36), -1)
    
    # Title
    cv2.putText(frame, "Video Smile Tracking", (20, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Stats
    y_offset = 65
    cv2.putText(frame, f"Frame: {frame_idx}/{total_frames}", (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    y_offset += 25
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    y_offset += 25
    total_people = stats.get('total_people', 0)
    cv2.putText(frame, f"Total People: {total_people}", (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    y_offset += 30
    people_smiling = stats.get('people_smiling', 0)
    cv2.putText(frame, f"People Smiling: {people_smiling}", (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


def process_video(args: argparse.Namespace) -> None:
    """Xử lý video với tracking."""
    if not args.video.exists():
        raise FileNotFoundError(f"Không tìm thấy video: {args.video}")
    
    print(f"🎬 Xử lý video: {args.video}")
    print(f"📊 Processing every {args.process_every} frames")
    print(f"🎯 Device: {args.device}")
    
    # Khởi tạo pipeline
    config = SmileCounterConfig(
        face_model=str(args.face_model),
        classifier_weights=str(args.weights),
        device=args.device,
    )
    counter = SmileCounter(config)
    
    # Khởi tạo tracker
    tracker = SimpleFaceTracker(
        iou_threshold=0.3,
        max_age=30,  # 1 giây (30 FPS)
        min_hits=3,
        distance_threshold=150
    )
    
    # Mở video
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise ValueError("Không thể mở video")
    
    # Video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"📹 Video info: {width}x{height} @ {fps} FPS, {total_frames} frames")
    
    # Video writer
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(args.output), fourcc, fps, (width, height))
    
    # Processing
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
            
            # Phát hiện scene change
            if prev_frame is not None:
                is_scene_change = detect_scene_change(prev_frame, frame, args.scene_threshold)
                if is_scene_change:
                    print(f"\n🔄 Scene change detected at frame {frame_idx}, resetting tracker")
                    tracker.reset()
                    last_detections = []
            
            # Chỉ detect ở một số frames
            if frame_idx % args.process_every == 0:
                # Phát hiện và phân loại
                summary = counter.analyze_array(frame)
                detections = summary.get('detections', [])
                
                # Chuyển format cho tracker
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
                # Dùng detections từ frame trước
                tracker_detections = last_detections
            
            # Update tracker
            confirmed_tracks = tracker.update(tracker_detections)
            
            # Vẽ tracked faces
            for track in confirmed_tracks:
                draw_tracked_face(
                    frame,
                    track.track_id,
                    track.bbox,
                    track.smile_probability,
                    track.is_smiling,
                    track.get_smile_ratio()
                )
            
            # Tính FPS
            processing_time = time.time() - start_time
            processing_times.append(processing_time)
            if len(processing_times) > 30:
                processing_times.pop(0)
            avg_fps = 1.0 / (sum(processing_times) / len(processing_times))
            
            # Lấy statistics
            stats = tracker.get_statistics()
            
            # Vẽ statistics
            draw_statistics(frame, stats, avg_fps, frame_idx, total_frames)
            
            # Ghi frame
            out.write(frame)
            
            prev_frame = frame.copy()
            pbar.update(1)
    
    finally:
        cap.release()
        out.release()
        pbar.close()
    
    # Final statistics
    print("\n" + "="*60)
    print("📊 Final Statistics")
    print("="*60)
    
    final_stats = tracker.get_statistics()
    print(f"Total People Tracked: {final_stats['total_people']}")
    print(f"People Smiling: {final_stats['people_smiling']}")
    print()
    
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
    args = parse_args()
    process_video(args)


if __name__ == "__main__":
    main()

"""Ví dụ chạy nhanh cho pipeline SmileCounter."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import torch

from src.pipeline.smile_counter import SmileCounter, SmileCounterConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phát hiện người cười trong ảnh nhóm")
    parser.add_argument("image", type=Path, help="Đường dẫn ảnh cần phân tích")
    parser.add_argument("--output", type=Path, default=Path("annotated.jpg"), help="Ảnh xuất chú thích")
    parser.add_argument("--weights", type=Path, default=Path("models/smile_cnn_best.pth"), help="Trọng số classifier")
    parser.add_argument("--face-model", type=str, default="models/yolov8n-face.pt", help="Model YOLO face")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def run_demo(args: argparse.Namespace) -> None:
    if not args.image.exists():
        raise FileNotFoundError(f"Không tìm thấy ảnh: {args.image}")
    image = cv2.imread(str(args.image))
    if image is None:
        raise ValueError("Không đọc được ảnh")

    config = SmileCounterConfig(
        face_model=str(args.face_model),
        classifier_weights=str(args.weights),
        device=args.device,
    )
    counter = SmileCounter(config)
    summary = counter.analyze_array(image)

    print("Tổng mặt:", summary["total_faces"])
    print("Số người cười:", summary["smiling_faces"])
    for idx, item in enumerate(summary["detections"], start=1):
        trang_thai = "cười" if item["is_smiling"] else "không"
        print(f" - Mặt {idx}: {trang_thai}, xác suất {item['smile_probability']:.2f}")

    annotated = counter.annotate(image, summary["detections"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), annotated)
    print(f"Đã lưu ảnh gắn nhãn tại {args.output}")


def main() -> None:
    run_demo(parse_args())


if __name__ == "__main__":
    main()

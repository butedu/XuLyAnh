"""Ví dụ chạy nhanh cho pipeline SmileCounter."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import torch

# Import pipeline đếm nụ cười
from src.pipeline.smile_counter import SmileCounter, SmileCounterConfig


def parse_args() -> argparse.Namespace:
    # Tạo bộ parser để nhận tham số dòng lệnh
    parser = argparse.ArgumentParser(description="Phát hiện người cười trong ảnh nhóm")

    # Ảnh cần phân tích (bắt buộc)
    parser.add_argument("image", type=Path, help="Đường dẫn ảnh cần phân tích")

    # File ảnh output sau khi annotate
    parser.add_argument("--output", type=Path, default=Path("annotated.jpg"), help="Ảnh xuất chú thích")

    # Trọng số của classifier mô hình SmileNet
    parser.add_argument("--weights", type=Path, default=Path("models/smile_cnn_best.pth"), help="Trọng số classifier")

    # File mô hình YOLO dùng để detect khuôn mặt
    parser.add_argument("--face-model", type=str, default="models/yolov8n-face.pt", help="Model YOLO face")

    # Chọn device, ưu tiên CUDA nếu có
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()


def run_demo(args: argparse.Namespace) -> None:
    # Kiểm tra ảnh có tồn tại không
    if not args.image.exists():
        raise FileNotFoundError(f"Không tìm thấy ảnh: {args.image}")

    # Đọc ảnh với OpenCV
    image = cv2.imread(str(args.image))
    if image is None:
        raise ValueError("Không đọc được ảnh")

    # Tạo cấu hình cho SmileCounter
    config = SmileCounterConfig(
        face_model=str(args.face_model),
        classifier_weights=str(args.weights),
        device=args.device,
    )

    # Khởi tạo pipeline đếm nụ cười
    counter = SmileCounter(config)

    # Phân tích ảnh dạng numpy array
    summary = counter.analyze_array(image)

    # In tổng kết
    print("Tổng mặt:", summary["total_faces"])
    print("Số người cười:", summary["smiling_faces"])

    # In từng khuôn mặt và xác suất cười
    for idx, item in enumerate(summary["detections"], start=1):
        trang_thai = "cười" if item["is_smiling"] else "không"
        print(f" - Mặt {idx}: {trang_thai}, xác suất {item['smile_probability']:.2f}")

    # Annotate ảnh (vẽ bounding box + label)
    annotated = counter.annotate(image, summary["detections"])

    # Tạo thư mục nếu chưa có rồi lưu file output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), annotated)

    print(f"Đã lưu ảnh gắn nhãn tại {args.output}")


def main() -> None:
    # Gọi hàm demo với tham số dòng lệnh
    run_demo(parse_args())


# Entry point khi chạy file trực tiếp
if __name__ == "__main__":
    main()

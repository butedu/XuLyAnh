"""Chạy nhận diện nụ cười trên video hoặc webcam."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import cv2
import torch

# Pipeline nhận diện + đếm nụ cười
from src.pipeline.smile_counter import SmileCounter, SmileCounterConfig


def build_argparser() -> argparse.ArgumentParser:
    # Tạo parser để đọc tham số dòng lệnh
    parser = argparse.ArgumentParser(description="Phát hiện nụ cười trong video")

    # Nguồn video: file path hoặc "0" để mở webcam
    parser.add_argument("source", help="Đường dẫn video đầu vào hoặc 0 để dùng webcam")

    # Đường dẫn lưu video đã annotate
    parser.add_argument(
        "--output",
        type=Path,
        help="Đường dẫn lưu video đã chú thích (bỏ qua nếu chỉ xem trực tiếp)",
    )

    # Model YOLO detect khuôn mặt
    parser.add_argument(
        "--face-model",
        type=str,
        default="models/yolov8n-face.pt",
        help="Đường dẫn hoặc tên model YOLO face",
    )

    # Trọng số classifier SmileNet
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("models/smile_cnn_best.pth"),
        help="Trọng số bộ phân loại nụ cười",
    )

    # Chọn thiết bị chạy
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Thiết bị sử dụng (cuda/cpu)",
    )

    # Hiển thị preview realtime
    parser.add_argument(
        "--display",
        action="store_true",
        help="Hiển thị cửa sổ preview trong quá trình xử lý",
    )

    # Bỏ qua frame để tăng tốc độ xử lý
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=0,
        help="Số frame bỏ qua giữa các lần suy luận (0 = xử lý mọi frame)",
    )

    # Resize khung hình trước khi xử lý (tăng tốc hoặc chuẩn hóa input)
    parser.add_argument(
        "--resize",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        help="Resize khung hình trước khi phân tích (ví dụ: --resize 1280 720)",
    )
    return parser


def open_capture(source: str) -> cv2.VideoCapture:
    # Nếu là số => webcam, còn lại => file video
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
    else:
        cap = cv2.VideoCapture(source)

    # Kiểm tra mở được hay không
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được nguồn video: {source}")

    return cap


def create_writer(cap: cv2.VideoCapture, output_path: Path) -> cv2.VideoWriter:
    # FourCC cho file mp4
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # Lấy FPS từ nguồn, fallback = 25 nếu không có
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    # Lấy kích thước khung hình
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Tạo writer ghi từng frame output
    return cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))


def overlay_summary(frame, total: int, smiles: int) -> None:
    # Vẽ hộp thông tin tổng số mặt và số người cười
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


def process_video(
    source: str,
    output: Optional[Path],
    config: SmileCounterConfig,
    display: bool,
    frame_skip: int,
    resize_dims: Optional[tuple[int, int]],
) -> None:
    # Mở video/webcam
    cap = open_capture(source)

    writer: Optional[cv2.VideoWriter] = None

    # Nếu có yêu cầu lưu file => tạo writer
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        writer = create_writer(cap, output)

    # Khởi tạo pipeline
    counter = SmileCounter(config)

    frame_index = 0

    try:
        while True:
            grabbed, frame = cap.read()
            if not grabbed:
                break  # Hết video

            # Resize nếu người dùng truyền ---resize
            if resize_dims is not None:
                frame = cv2.resize(frame, resize_dims)

            # Xử lý frame theo frame_skip
            if frame_skip and frame_index % (frame_skip + 1) != 0:
                # Không chạy detect => giữ nguyên frame
                annotated = frame
            else:
                # Phát hiện và phân loại nụ cười
                summary = counter.analyze_array(frame)

                # Annotate bounding box + label
                annotated = counter.annotate(frame, summary["detections"])

                # Vẽ overlay tóm tắt
                overlay_summary(annotated, summary["total_faces"], summary["smiling_faces"])

            # Nếu có output => ghi frame ra file
            if writer is not None:
                writer.write(annotated)

            # Hiển thị realtime
            if display:
                cv2.imshow("SmileCounter", annotated)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC để thoát
                    break

            frame_index += 1

    finally:
        # Giải phóng tài nguyên
        cap.release()
        if writer is not None:
            writer.release()
        if display:
            cv2.destroyAllWindows()


def main() -> None:
    # Lấy tham số dòng lệnh
    parser = build_argparser()
    args = parser.parse_args()

    # Tạo config cho pipeline
    config = SmileCounterConfig(
        face_model=args.face_model,
        classifier_weights=str(args.weights),
        device=args.device,
    )

    # Nếu có resize => chuyển sang tuple
    resize_dims = tuple(args.resize) if args.resize else None

    # Chạy xử lý video
    process_video(
        source=args.source,
        output=args.output,
        config=config,
        display=args.display,
        frame_skip=max(args.frame_skip, 0),
        resize_dims=resize_dims,
    )


if __name__ == "__main__":
    main()

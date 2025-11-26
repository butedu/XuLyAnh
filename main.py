"""
VÍ DỤ SỬ DỤNG: Phát hiện nụ cười trong ảnh nhóm

Script này minh họa cách sử dụng bộ phát hiện nụ cười để:
1. Tải ảnh từ file
2. Phát hiện tất cả khuôn mặt
3. Phân loại cười/không cười cho từng khuôn mặt
4. Vẽ khung và nhãn lên ảnh
5. Lưu kết quả
"""

import sys
from pathlib import Path

# Thêm thư mục gốc vào Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
from src.inference.smile_detector import BoNhanDienCuoi, CauHinhBoNhanDienCuoi


def phat_hien_cuoi_trong_anh(duong_dan_anh: str, duong_dan_luu: str) -> None:
    """
    Phát hiện nụ cười trong ảnh và lưu kết quả.

    Tham số:
        duong_dan_anh: Đường dẫn đến ảnh đầu vào
        duong_dan_luu: Đường dẫn để lưu ảnh kết quả
    """
    print(f"Đang phân tích ảnh: {duong_dan_anh}")

    # Bước 1: Khởi tạo bộ phát hiện
    # Sử dụng Haar cascade (nhanh) hoặc DNN (chính xác hơn)
    cau_hinh = CauHinhBoNhanDienCuoi(
        face_backend="haar",  # Có thể đổi thành "dnn" nếu có model
        model_weights="models/smile_cnn_best.pth",
    )
    bo_phat_hien = BoNhanDienCuoi(cau_hinh)

    # Bước 2: Đọc ảnh
    anh = cv2.imread(duong_dan_anh)
    if anh is None:
        print(f"Lỗi: Không thể đọc ảnh từ {duong_dan_anh}")
        return

    # Bước 3: Phân tích ảnh
    print("Đang phát hiện khuôn mặt và phân loại nụ cười...")
    ket_qua = bo_phat_hien.phan_tich(anh)

    # Bước 4: In kết quả
    print("\n" + "=" * 50)
    print("KẾT QUẢ PHÂN TÍCH")
    print("=" * 50)
    print(f"Tổng số khuôn mặt: {ket_qua['total_faces']}")
    print(f"Số người đang cười: {ket_qua['smiling_faces']}")
    print(f"Số người không cười: {ket_qua['total_faces'] - ket_qua['smiling_faces']}")

    # In chi tiết từng khuôn mặt
    print("\nChi tiết từng khuôn mặt:")
    for i, phat_hien in enumerate(ket_qua['detections'], 1):
        trang_thai = "Đang cười" if phat_hien['is_smiling'] else "Bình thường"
        xac_suat = phat_hien['smile_probability']
        print(f"  Khuôn mặt {i}: {trang_thai} (Xác suất: {xac_suat:.2%})")

    # Bước 5: Vẽ chú thích lên ảnh
    print("\nĐang vẽ chú thích...")
    anh_ket_qua = bo_phat_hien.chu_thich_anh(anh, ket_qua['detections'])

    # Bước 6: Lưu ảnh kết quả
    cv2.imwrite(duong_dan_luu, anh_ket_qua)
    print(f"Đã lưu ảnh kết quả tại: {duong_dan_luu}")
    print("=" * 50 + "\n")


def main() -> None:
    """Hàm chính."""
    # Ví dụ sử dụng
    duong_dan_anh = "data/test/group_photo.jpg"
    duong_dan_ket_qua = "data/test/result_annotated.jpg"

    # Kiểm tra file tồn tại
    if not Path(duong_dan_anh).exists():
        print(f"Lỗi: Không tìm thấy file ảnh tại {duong_dan_anh}")
        print("Vui lòng cập nhật đường dẫn trong script hoặc đặt ảnh test vào thư mục data/test/")
        return

    # Chạy phát hiện
    phat_hien_cuoi_trong_anh(duong_dan_anh, duong_dan_ket_qua)


if __name__ == "__main__":
    main()

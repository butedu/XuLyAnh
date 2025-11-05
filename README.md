# XuLyAnh

Đề tài: Phát hiện nụ cười / Đếm số người cười trong ảnh nhóm
Mục tiêu
Từ một ảnh nhóm, chương trình sẽ:
Phát hiện khuôn mặt (face detection)
Phân loại mỗi khuôn mặt là “cười” hay “không cười” (smile classification)
Đếm và hiển thị số người đang cười trên ảnh

# Kiến thức bạn cần nắm

OpenCV cơ bản: đọc ảnh, hiển thị, vẽ hình chữ nhật.
Phát hiện khuôn mặt bằng Haar Cascade hoặc DNN.
Huấn luyện/áp dụng mô hình CNN (Keras/TensorFlow hoặc PyTorch).
Tiền xử lý dữ liệu ảnh: resize, normalization.
Xử lý kết quả, đếm và hiển thị thống kê.

# Các bước thực hiện

Bước 1 – Chuẩn bị môi trường làm việc:
Cài đặt Python và các thư viện cần thiết:
pip install opencv-python opencv-contrib-python tensorflow keras numpy matplotlib

Tạo cấu trúc dự án:
XULYANH/
│
├── data/                # chứa dữ liệu huấn luyện (GENKI-4K, RAF-DB)
├── models/              # lưu mô hình sau khi huấn luyện
├── src/                 # mã nguồn chính
│   ├── train_smile_model.py     # huấn luyện CNN
│   ├── detect_and_classify.py   # chương trình chính phát hiện & đếm nụ cười
│   └── utils.py                 # các hàm phụ trợ
└── requirements.txt     # danh sách thư viện cần cài

Bước 2 – Phát hiện khuôn mặt
Bước 3 – Huấn luyện bộ phân loại nụ cười (Smile Classifier)
Bước 4 – Phát hiện & phân loại nụ cười trong ảnh nhóm
Bước 5 – Đánh giá và mở rộng

Tổng kết quy trình
Bước	      Mô tả	                                Kết quả
1	    Chuẩn bị môi trường và dataset	   Có môi trường chạy code
2	    Phát hiện khuôn mặt	               Nhận được tọa độ từng khuôn mặt
3	    Huấn luyện mô hình CNN	           File smile_detector.h5
4	    Chạy chương trình chính	           Đếm và hiển thị số người cười
5	    Đánh giá và cải tiến	           Báo cáo kết quả, tối ưu mô hình
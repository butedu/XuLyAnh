pip install opencv-python opencv-contrib-python tensorflow keras numpy matplotlib
## Giới thiệu
Dự án cung cấp pipeline hoàn chỉnh cho bài toán phát hiện nụ cười và đếm số người cười trong ảnh nhóm. Hệ thống gồm các phần:
- Tiền xử lý và chuẩn bị dữ liệu từ GENKI-4K và RAF-DB
- Huấn luyện bộ phân loại nụ cười dựa trên CNN nhẹ
- Pipeline suy luận kết hợp phát hiện khuôn mặt (Haar cascade hoặc DNN) và phân loại nụ cười
- Ứng dụng web (FastAPI + HTML/JS) để demo tải ảnh, nhận kết quả và xem ảnh chú thích

## Cấu trúc chính
```
XuLyAnh/
├── requirements.txt
├── src/
│   ├── data/
│   │   ├── prepare_genki.py
│   │   └── prepare_rafdb.py
│   ├── inference/
│   │   └── smile_detector.py
│   ├── models/
│   │   ├── face_detector.py
│   │   └── smile_cnn.py
│   └── training/
│       ├── datasets.py
│       ├── train_smile_classifier.py
│       └── utils.py
├── webapp/
│   ├── backend/
│   │   ├── main.py
│   │   └── service.py
│   └── frontend/
│       └── index.html
└── models/
	 └── (lưu checkpoint sau huấn luyện)
```

## Thiết lập môi trường
1. Cài Python ≥ 3.10
2. Tạo môi trường ảo và cài thư viện:
	```bash
	python -m venv .venv
	.\.venv\Scripts\activate
	pip install -r requirements.txt
	```

## Chuẩn bị dữ liệu
### GENKI-4K
- Tải dataset từ trang GENKI (định dạng `.zip`) và giải nén, ví dụ `data/raw/genki4k/`
- Chạy script để tạo file CSV và chia tập:
  ```bash
  python -m src.data.prepare_genki --raw-dir data/raw/genki4k --output-dir data/processed/genki4k --create-splits
  ```
- Kết quả: `annotations.csv`, `train.csv`, `val.csv`, `test.csv`

### RAF-DB
- Đăng ký và tải bộ dữ liệu RAF-DB, giải nén về `data/raw/rafdb/`
- Tạo CSV:
  ```bash
  python -m src.data.prepare_rafdb --raw-dir data/raw/rafdb --output-dir data/processed/rafdb
  ```
- Script sẽ gộp nhãn “happiness” thành lớp cười và các biểu cảm khác thành không cười.

> **Lưu ý**: Các script giả định cột `filepath` là đường dẫn tương đối tính từ thư mục ảnh gốc. Nếu cấu trúc khác, chỉnh sửa CSV cho phù hợp.

## Huấn luyện mô hình
1. Gom tất cả ảnh sử dụng chung vào ví dụ `data/images/` và cập nhật các file CSV tương ứng.
2. Chạy huấn luyện (ví dụ dùng GENKI-4K):
	```bash
	python -m src.training.train_smile_classifier --image-root data/images --split-dir data/processed/genki4k --output-dir models
	```
3. Checkpoint tốt nhất được lưu tại `models/smile_cnn_best.pth` cùng lịch sử `training_history.json`.

## Suy luận và kiểm thử nhanh
Sử dụng module `SmileDetector`:
```python
from src.inference.smile_detector import SmileDetector, SmileDetectorConfig

config = SmileDetectorConfig(model_weights="models/smile_cnn_best.pth", face_backend="haar")
detector = SmileDetector(config)
results = detector.analyze_image("path/to/group_photo.jpg")
print(results)
```

Để chuyển sang detector DNN, tải file `deploy.prototxt` và `res10_300x300_ssd_iter_140000.caffemodel` từ OpenCV, đặt dưới `models/face_detector/` rồi đặt `face_backend="dnn"`.

## Chạy demo web
1. Đảm bảo đã có file `models/smile_cnn_best.pth`
2. Khởi động backend FastAPI:
	```bash
	uvicorn webapp.backend.main:app --reload --port 8000
	```
3. Mở trình duyệt tại `http://localhost:8000` và tải ảnh để xem kết quả.

API trả về JSON gồm tổng số khuôn mặt, số người cười và ảnh đã chú thích (base64). Giao diện web sẽ hiển thị trực quan.

## Gợi ý mở rộng
- Dùng augmentation mạnh hơn (Albumentations)
- Thử backbone tiền huấn luyện (ResNet, MobileNet)
- Fine-tune riêng cho từng dataset và ensemble
- Thêm đánh giá thống kê (confusion matrix, ROC)
- Đóng gói mô hình bằng TorchScript hoặc ONNX để triển khai thực tế
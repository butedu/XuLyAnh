## Tổng quan
Pipeline được viết lại hoàn toàn để chỉ dùng **YOLOv8-face** cho phát hiện khuôn mặt và **SmileNet (CNN nhẹ)** cho phân loại cười/không cười. Mục tiêu chính: đếm nhanh số người đang cười trong ảnh nhóm và sinh ảnh chú thích trực quan.

## Thành phần chính
- Chuẩn bị dữ liệu từ GENKI-4K, RAF-DB (`src/data/...`).
- Mô hình SmileNet và công cụ huấn luyện (`src/classifier`, `src/training`).
- Pipeline suy luận mới trong `src/pipeline/smile_counter.py`.
- Ứng dụng web FastAPI tại `webapp/backend` và giao diện HTML tĩnh.

```
XuLyAnh/
├── main.py                  # Ví dụ CLI
├── models/                  # Lưu trọng số: smile_cnn_best.pth, yolov8n-face.pt
├── src/
│   ├── classifier/          # SmileNet + helper
│   ├── data/                # Chuẩn bị dữ liệu gốc
│   ├── detection/           # YOLO face wrapper
│   ├── pipeline/            # SmileCounter
│   └── training/            # Huấn luyện + DataLoader
└── webapp/                  # Ứng dụng web demo
```

## Cài đặt nhanh
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Tải model YOLOv8-face (ví dụ `yolov8n-face.pt`) và đặt vào thư mục `models/`. Nếu bạn dùng tên khác, cập nhật khi khởi động pipeline.

## Chuẩn bị dữ liệu
### GENKI-4K
```powershell
python -m src.data.prepare_genki --raw-dir data/raw/genki4k --output-dir data/processed/genki4k --create-splits
```

### RAF-DB
```powershell
python -m src.data.prepare_rafdb --raw-dir data/raw/rafdb --output-dir data/processed/rafdb
```

Các CSV sinh ra phải có cột `filepath` (đường dẫn tương đối tới ảnh trong `--image-root`) và `label` (0=không cười, 1=cười).

## Huấn luyện SmileNet
```powershell
python -m src.training.train --image-root data/images --split-dir data/processed/genki4k --output-dir models
```

- `models/smile_cnn_best.pth`: checkpoint tốt nhất (theo F1).
- `models/training_history.json`: lịch sử loss/metric.
- Lệnh cũ `python -m src.training.train_smile_classifier` vẫn hoạt động và sẽ chuyển tiếp sang lệnh mới.

## Suy luận trên ảnh tĩnh
```powershell
python main.py data/test/group_photo.jpg --output data/test/group_photo_marked.jpg
```

Script sẽ in thống kê ra console và lưu ảnh chú thích với khung màu xanh (cười) hoặc đỏ (không cười).

## Chạy web demo
```powershell
uvicorn webapp.backend.main:app --reload --port 8000
```

Mặc định backend tìm `models/smile_cnn_best.pth` và `models/yolov8n-face.pt`. Mở trình duyệt tới `http://localhost:8000` để tải ảnh và xem kết quả.

## Ghi chú
- Nếu thiếu `ultralytics`, hãy cài thêm `pip install ultralytics`.
- Với GPU, bảo đảm CUDA tương thích cùng bản `torch` trong `requirements.txt`.
- Hãy cân nhắc augment thêm hoặc fine-tune bằng transfer learning nếu dữ liệu lớn.
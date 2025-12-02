# HƯỚNG DẪN SỬ DỤNG

## 1. Tổng quan
- Bài toán: phát hiện khuôn mặt trong ảnh nhóm → phân loại cười/không cười → đếm và chú thích.
- Phát hiện mặt: **YOLOv8-face** (ultralytics).
- Phân loại nụ cười: **SmileNet** (CNN nhẹ 64x64).

Pipeline rút gọn:
```
Ảnh đầu vào → YOLO phát hiện mặt → Cắt mặt (mở rộng nhẹ) → SmileNet dự đoán → Thống kê + vẽ kết quả
```

## 2. Cài đặt nhanh
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

- Tải `yolov8n-face.pt` hoặc model tương đương đặt vào `models/`.
- Nếu cần GPU, bảo đảm phiên bản CUDA phù hợp với `torch` trong yêu cầu.

## 3. Chuẩn bị dữ liệu
### GENKI-4K
```powershell
python -m src.data.prepare_genki --raw-dir data/raw/genki4k --output-dir data/processed/genki4k --create-splits
```
Sinh ra `train.csv`, `val.csv`, `test.csv` với hai cột `filepath`, `label`.

### RAF-DB (tùy chọn)
```powershell
python -m src.data.prepare_rafdb --raw-dir data/raw/rafdb --output-dir data/processed/rafdb
```
Lưu ý: nhãn "happiness" → cười, các nhãn khác → không cười.

### Gom ảnh
Đặt toàn bộ ảnh được tham chiếu trong CSV vào cùng một thư mục (ví dụ `data/images/`). Các đường dẫn trong CSV là tương đối so với thư mục này.

## 4. Huấn luyện SmileNet
```powershell
python -m src.training.train --image-root data/images --split-dir data/processed/genki4k --output-dir models
```

Tùy chỉnh thêm:
```powershell
python -m src.training.train `
    --image-root data/images `
    --split-dir data/processed/genki4k `
    --output-dir models `
    --epochs 30 `
    --batch-size 64 `
    --learning-rate 1e-3 `
    --device cuda
```

Kết quả lưu trong `models/`:
- `smile_cnn_best.pth`: trọng số tốt nhất (đo bằng F1 trên tập val).
- `checkpoint.pt`: đầy đủ trạng thái optimizer/scheduler.
- `training_history.json`: log từng epoch.

> Lệnh cũ `python -m src.training.train_smile_classifier` vẫn chạy và sẽ chuyển hướng sang lệnh mới.

## 5. Suy luận nhanh bằng CLI
```powershell
python main.py data/test/group_photo.jpg --output data/test/group_photo_marked.jpg
```
- In thống kê số người cười ra console.
- Lưu ảnh gắn nhãn (khung xanh = cười, đỏ = không cười).

Các tuỳ chọn hữu ích:
- `--weights`: đường dẫn trọng số SmileNet khác.
- `--face-model`: tên/đường dẫn model YOLO khác.
- `--device`: `cpu` hoặc `cuda`.

## 6. Dùng trong FastAPI
Khởi động backend demo:
```powershell
uvicorn webapp.backend.main:app --reload --port 8000
```

- Backend tự tìm `models/smile_cnn_best.pth` và `models/yolov8n-face.pt` tại thư mục gốc dự án.
- Mở trình duyệt: `http://localhost:8000` → tải ảnh → xem kết quả live.

## 7. Cấu trúc mã nguồn mới
- `src/detection/yolo.py`: lớp `YOLOFaceDetector` (Ultralytics).
- `src/classifier/smile_model.py`: kiến trúc SmileNet và hàm build/load.
- `src/pipeline/smile_counter.py`: ghép các bước phát hiện + phân loại.
- `src/training/data.py`: dataset, DataLoader, chia tách CSV.
- `src/training/train.py`: vòng lặp huấn luyện chính.
- `webapp/backend/service.py`: dùng `SmileCounter` cho API.

Các file cũ dùng Haar/DNN hoặc MTCNN đã được loại bỏ hoàn toàn.

## 8. Mẹo và mở rộng
- **Tăng độ chính xác:** thử `ColorJitter`, `RandomRotation`, hoặc fine-tune backbone khác (ResNet18, MobileNetV3).
- **Giảm nhiễu:** điều chỉnh `expand_ratio` trong `SmileCounterConfig` để cân bằng vùng miệng.
- **Tối ưu tốc độ:** giảm `max_faces` của YOLO hoặc chạy trên GPU.
- **Triển khai:** xuất TorchScript/ONNX từ trọng số nếu cần môi trường production.

Nếu bạn cần thêm dataset khác, hãy chuẩn bị CSV cùng định dạng `filepath,label` và tái sử dụng các script hiện có.

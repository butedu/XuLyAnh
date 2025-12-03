# Hệ Thống Nhận Diện Nụ Cười

Hệ thống phát hiện khuôn mặt và phân loại nụ cười trong ảnh sử dụng **YOLOv8-face** và **SmileNet (CNN residual)**. Hỗ trợ xử lý ảnh tĩnh, video, và cung cấp API web để demo trực tuyến.

---

## 📋 Tổng Quan Hệ Thống

### Kiến Trúc
```
Ảnh đầu vào → YOLOv8-face (phát hiện mặt) → SmileNet (phân loại cười/không) → Thống kê + Chú thích
```

### Thành Phần Chính
- **YOLOv8-face**: Phát hiện khuôn mặt nhanh, chính xác (Ultralytics)
- **SmileNet**: Mạng CNN residual 7 stage với backbone sâu, dropout regularization
- **SmileCounter**: Pipeline tích hợp phát hiện + phân loại + vẽ chú thích
- **FastAPI Backend**: RESTful API cho web demo và xử lý video
- **CLI Tool**: Script `main.py` cho xử lý ảnh nhanh từ command line

### Dataset
- **GENKI-4K**: 4000 ảnh khuôn mặt được gán nhãn cười/không cười (dataset chính)
- **RAF-DB**: (Tùy chọn) Dataset cảm xúc mở rộng, có thể dùng để tăng đa dạng dữ liệu

### Cấu Trúc Thư Mục
```
XuLyAnh/
├── main.py                      # CLI xử lý ảnh đơn
├── video_demo.py                # CLI xử lý video
├── requirements.txt             # Dependencies Python
├── models/                      # Trọng số đã train
│   ├── smile_cnn_best.pth      # SmileNet checkpoint tốt nhất
│   ├── yolov8n-face.pt         # YOLOv8-face pretrained
│   ├── checkpoint.pt           # Full checkpoint (optimizer, scheduler)
│   └── training_history.json   # Lịch sử huấn luyện
├── data/
│   ├── images/                 # Ảnh GENKI-4K (raw)
│   └── processed/              # CSV annotations đã xử lý
│       └── genki4k/
│           ├── train.csv
│           ├── val.csv
│           └── test.csv
├── src/
│   ├── classifier/             # SmileNet architecture
│   ├── detection/              # YOLOv8 wrapper
│   ├── pipeline/               # SmileCounter pipeline
│   ├── training/               # Training loop, datasets
│   └── data/                   # Data preprocessing scripts
└── webapp/
    ├── backend/                # FastAPI server
    └── frontend/               # HTML/CSS/JS interface
```

---

## 🚀 Cài Đặt & Chạy Chương Trình

### Bước 1: Clone Repository
```powershell
git clone <repository-url>
cd XuLyAnh
```

### Bước 2: Tạo Môi Trường Ảo
```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

### Bước 3: Cài Đặt Dependencies
```powershell
pip install -r requirements.txt
```

**Lưu ý GPU**: Nếu dùng GPU, đảm bảo CUDA toolkit tương thích với PyTorch 2.2.1:
```powershell
# Kiểm tra CUDA version
nvidia-smi

# Cài PyTorch với CUDA (nếu cần)
pip install torch==2.2.1 torchvision==0.17.1 --index-url https://download.pytorch.org/whl/cu118
```

### Bước 4: Tải Model YOLOv8-face
Tải pretrained model `yolov8n-face.pt` và đặt vào thư mục `models/`:
```powershell
# Tạo thư mục models nếu chưa có
New-Item -ItemType Directory -Force -Path models

# Tải model (ví dụ từ Ultralytics hoặc nguồn khác)
# Đặt file yolov8n-face.pt vào models/
```

### Bước 5: Chuẩn Bị Dữ Liệu GENKI-4K

**5.1. Tải Dataset**
- Tải GENKI-4K từ nguồn chính thức
- Giải nén vào `data/raw/genki4k/`

**5.2. Xử Lý Annotations**
```powershell
python -m src.data.prepare_genki `
    --raw-dir data/raw/genki4k `
    --output-dir data/processed/genki4k `
    --create-splits
```

Lệnh này sẽ tạo:
- `data/processed/genki4k/train.csv` (80%)
- `data/processed/genki4k/val.csv` (10%)
- `data/processed/genki4k/test.csv` (10%)

**5.3. Gom Ảnh**
Copy toàn bộ ảnh GENKI-4K vào `data/images/` để training script đọc được.

### Bước 6: Huấn Luyện Mô Hình (Tùy Chọn)

**Nếu đã có checkpoint:** Bỏ qua bước này, dùng `models/smile_cnn_best.pth` sẵn có.

**Huấn luyện từ đầu:**
```powershell
python -m src.training.train `
    --image-root data/images `
    --split-dir data/processed/genki4k `
    --output-dir models `
    --epochs 50 `
    --batch-size 64 `
    --learning-rate 5e-4 `
    --device cuda
```

**Resume từ checkpoint cũ:**
```powershell
python -m src.training.train `
    --image-root data/images `
    --split-dir data/processed/genki4k `
    --output-dir models `
    --epochs 60 `
    --resume models/checkpoint.pt `
    --device cuda
```

Kết quả:
- `models/smile_cnn_best.pth`: Trọng số tốt nhất (theo F1 score)
- `models/checkpoint.pt`: Full state (optimizer, scheduler)
- `models/training_history.json`: Metrics theo từng epoch

---

## 💻 Sử Dụng Hệ Thống

### 1. Xử Lý Ảnh Đơn (CLI)
```powershell
python main.py path/to/image.jpg --output result.jpg
```

**Tùy chọn:**
- `--weights`: Đường dẫn checkpoint khác (mặc định `models/smile_cnn_best.pth`)
- `--face-model`: Đường dẫn YOLOv8 khác (mặc định `models/yolov8n-face.pt`)
- `--device`: `cpu` hoặc `cuda`

**Kết quả:**
- In ra console: số mặt phát hiện, số người cười, xác suất từng mặt
- Lưu ảnh chú thích với:
  - Khung **xanh lá**: đang cười
  - Khung **đỏ**: không cười
  - Text hiển thị xác suất

### 2. Web Demo (FastAPI)
```powershell
uvicorn webapp.backend.main:app --reload --port 8000
```

Mở trình duyệt: `http://localhost:8000`

**Chức năng:**
- Upload ảnh → Xem kết quả trực tiếp
- Tải ảnh chú thích về máy
- Thống kê realtime số người cười

### 3. Xử Lý Video (CLI)
```powershell
python video_demo.py path/to/video.mp4 --output output_video.mp4
```

Xử lý từng frame, ghi video có chú thích khuôn mặt và thống kê.

---

## 🔧 Cấu Hình Nâng Cao

### Tùy Chỉnh SmileNet
Chỉnh `src/classifier/smile_model.py` để:
- Thay đổi số stage residual
- Điều chỉnh dropout rate
- Thay đổi kích thước input (mặc định 64x64)

### Tùy Chỉnh Pipeline
Chỉnh `src/pipeline/smile_counter.py`:
- `smile_threshold`: Ngưỡng xác suất coi là cười (mặc định 0.5)
- `expand_ratio`: Tỷ lệ mở rộng bounding box (mặc định 1.15)

### Augmentation
Chỉnh `src/training/train.py` → `build_transforms()`:
- Thêm `RandomRotation(5)` cho rotation
- Thêm `RandomErasing()` cho cutout
- Điều chỉnh `ColorJitter` parameters

---

## 📊 Hiệu Suất Mô Hình

**Training Results (GENKI-4K, 25 epochs):**
- Validation Accuracy: **88.25%**
- Validation F1 Score: **0.8878**
- Validation Precision: **91.63%**
- Validation Recall: **86.11%**

**Kiến trúc mới (Residual SmileNet):**
- 7 residual stages với dropout regularization
- Global average pooling thay MaxPool
- 2-layer classifier head
- Khởi tạo trọng số Kaiming/Xavier

---

## 🛠 Troubleshooting

### Lỗi "Không tìm thấy checkpoint"
- Kiểm tra file `models/smile_cnn_best.pth` và `models/yolov8n-face.pt` tồn tại
- Nếu chưa train, phải chạy bước huấn luyện trước

### Lỗi CUDA out of memory
- Giảm `--batch-size` xuống 32 hoặc 16
- Hoặc chuyển sang `--device cpu`

### Accuracy thấp
- Tăng số epoch (50-100)
- Giảm learning rate (`5e-4` hoặc `1e-4`)
- Thêm augmentation mạnh hơn
- Kết hợp RAF-DB để tăng đa dạng

### Web demo không load
- Kiểm tra `webapp/frontend/index.html` tồn tại
- Đảm bảo port 8000 không bị chiếm
- Xem log terminal để debug

---

## 📚 Tham Khảo

- **YOLOv8**: [Ultralytics Documentation](https://docs.ultralytics.com/)
- **GENKI-4K**: Dataset công khai cho smile detection
- **PyTorch**: [pytorch.org](https://pytorch.org/)
- **FastAPI**: [fastapi.tiangolo.com](https://fastapi.tiangolo.com/)


# 😊 Hệ Thống Nhận Diện Nụ Cười

Hệ thống phát hiện khuôn mặt và phân loại nụ cười sử dụng **YOLOv8-face** + **SmileNet CNN**.


## 🚀 Cài Đặt

### Yêu Cầu
- Python 3.10+
- GPU với CUDA (khuyến nghị) hoặc CPU

### Các Bước

```powershell
# 1. Clone repository
git clone https://github.com/butedu/XuLyAnh.git
cd XuLyAnh

# 2. Tạo môi trường ảo
python -m venv .venv
.\.venv\Scripts\activate

# 3. Cài đặt thư viện
pip install -r requirements.txt

# 4. Kiểm tra cài đặt
python test_setup.py
```

---

## 💻 Hướng Dẫn Sử Dụng

### 1. Xử Lý Ảnh Đơn

```powershell
python main.py path/to/image.jpg --output result.jpg
```

**Ví dụ:**
```powershell
# Xử lý ảnh nhóm
python main.py photos/team.jpg --output results/team_annotated.jpg

# Dùng CPU
python main.py image.jpg --device cpu
```

**Kết quả:**
- Console hiển thị: số mặt, số người cười, xác suất từng người
- Ảnh output: khung **xanh** = cười, khung **đỏ** = không cười

---

### 2. Xử Lý Video

#### a) Video đơn giản (không tracking)

```powershell
python video_demo.py video.mp4 --output output.mp4 --display
```


#### b) Video với Face Tracking (Khuyến nghị)

```powershell
python video_demo_tracking.py video.mp4 --output output_tracked.mp4
```

**Tính năng:**
- ✅ Gán ID cố định cho mỗi khuôn mặt
- ✅ Theo dõi xuyên suốt video
- ✅ Thống kê tỷ lệ cười từng người
- ✅ Tự động phát hiện chuyển cảnh



#### c) Webcam realtime

```powershell
python video_demo.py 0 --display
```

---

### 3. Web Demo

```powershell
# Khởi động server
uvicorn webapp.backend.main:app --host 127.0.0.1 --port 8000
```

Mở trình duyệt: **http://127.0.0.1:8000**

**Chức năng:**
- Upload ảnh → Xem kết quả trực tiếp
- Upload video → Xử lý với tracking
- Tải ảnh/video kết quả về máy

---

## 🎓 Huấn Luyện Model

### Quick Start

```powershell
# 1. Chuẩn bị dữ liệu GENKI-4K
python -m src.data.prepare_genki --raw-dir data/raw/genki4k --output-dir data/processed/genki4k --create-splits

# 2. Kiểm tra setup
python test_setup.py

# 3. Train
python train_model.py
```

### Tùy Chỉnh Training

Chỉnh sửa file `config/train_config.yaml`:

```yaml
training:
  epochs: 50
  batch_size: 64        # Giảm nếu hết RAM GPU
  learning_rate: 5.0e-4

model:
  name: "SmileNetV2"    # Hoặc "SmileNet" cho baseline
  use_se_block: true    # Bật SE attention

settings:
  use_amp: true         # Mixed precision (nhanh hơn 2-3x)
  device: "cuda"        # Hoặc "cpu"
```

### Resume Training

```powershell
python -m src.training.train --config config/train_config.yaml --resume models/checkpoint.pt
```

### Xem Kết Quả Training

```powershell
python visualize_training.py
```

Vẽ biểu đồ loss, accuracy, F1 score qua các epoch.

---

## 📁 Cấu Trúc Dự Án

```
XuLyAnh/
├── main.py                 # CLI xử lý ảnh
├── video_demo.py           # CLI xử lý video (không tracking)
├── video_demo_tracking.py  # CLI xử lý video (có tracking)
├── train_model.py          # Script huấn luyện
├── test_setup.py           # Kiểm tra môi trường
├── visualize_training.py   # Vẽ biểu đồ training
├── requirements.txt        # Dependencies
│
├── config/
│   └── train_config.yaml   # Cấu hình training
│
├── models/                 # Trọng số model
│   ├── smile_cnn_best.pth  # Model tốt nhất
│   ├── yolov8n-face.pt     # YOLO face detector
│   └── training_history.json
│
├── src/
│   ├── classifier/         # SmileNet architecture
│   ├── detection/          # YOLO wrapper
│   ├── pipeline/           # SmileCounter pipeline
│   ├── tracking/           # Face tracker
│   └── training/           # Training code
│
├── webapp/
│   ├── backend/            # FastAPI server
│   └── frontend/           # Web interface
│
└── data/
    ├── raw/                # Dataset gốc
    └── processed/          # CSV đã xử lý
```

---

## 🛠 Troubleshooting

### CUDA out of memory

```yaml
# Trong config/train_config.yaml
training:
  batch_size: 32    # Giảm từ 64
settings:
  use_amp: true     # Bật mixed precision
```

### Model không học / Accuracy thấp

1. Kiểm tra data cân bằng:
```powershell
python -c "import pandas as pd; print(pd.read_csv('data/processed/genki4k/train.csv')['label'].value_counts())"
```

2. Giảm learning rate và tăng epochs:
```yaml
training:
  epochs: 80
  learning_rate: 1.0e-4
```

### Lỗi "Không tìm thấy model"

Đảm bảo các file tồn tại:
- `models/smile_cnn_best.pth`
- `models/yolov8n-face.pt`

Nếu chưa có, cần [huấn luyện model](#-huấn-luyện-model) trước.

### Web demo không hoạt động

```powershell
# Kiểm tra port 8000 có bị chiếm không
netstat -ano | findstr :8000

# Thử port khác
uvicorn webapp.backend.main:app --port 8080
```

---

## 📊 Hiệu Suất

| Model | Accuracy | F1 Score | Parameters |
|-------|----------|----------|------------|
| SmileNet (baseline) | 88.25% | 0.8878 | ~500K |
| SmileNetV2 | 90-92% | 0.91-0.93 | ~750K |

---

## 📚 Tham Khảo

- [YOLOv8 - Ultralytics](https://docs.ultralytics.com/)
- [GENKI-4K Dataset](https://inc.ucsd.edu/mplab/398.php)
- [PyTorch](https://pytorch.org/)
- [FastAPI](https://fastapi.tiangolo.com/)


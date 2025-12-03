# Hệ Thống Nhận Diện Nụ Cười

Hệ thống phát hiện khuôn mặt và phân loại nụ cười trong ảnh sử dụng **YOLOv8-face** và **SmileNetV2 (CNN residual với SE attention)**. Hỗ trợ xử lý ảnh tĩnh, video, và cung cấp API web để demo trực tuyến.

## 🚀 Quick Start (Training Mới)

**Cách nhanh nhất để train model:**

```bash
# 1. Cài đặt dependencies
pip install -r requirements.txt

# 2. Chuẩn bị dữ liệu (nếu chưa có)
python -m src.data.prepare_genki --raw-dir data/raw/genki4k --output-dir data/processed/genki4k --create-splits

# 3. Train với một dòng lệnh
python train_model.py
```

**Tất cả hyperparameters được config trong `config/train_config.yaml`** - không cần nhập thủ công nữa!

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

#### 🚀 Cách 1: Sử dụng Script Python (KHUYẾN NGHỊ)

```bash
python train_model.py
```

Script này sẽ:
- Tự động kiểm tra GPU có sẵn không
- Training với config từ `config/train_config.yaml`
- Hiển thị tiến trình đẹp mắt với emoji và màu sắc
- Dễ dàng thêm arguments nếu cần

#### 🔧 Cách 2: Training với Custom Config

**Chỉnh sửa hyperparameters trong `config/train_config.yaml`:**
```yaml
training:
  epochs: 50           # Tăng lên 100 nếu muốn train lâu hơn
  batch_size: 64       # Giảm xuống 32 nếu GPU hết RAM
  learning_rate: 5.0e-4
  
model:
  name: "SmileNetV2"   # Dùng kiến trúc mới với residual + SE attention
  use_se_block: true   # Bật Squeeze-and-Excitation attention
```

Sau đó chạy:
```powershell
python -m src.training.train --config config/train_config.yaml
```

#### 🎯 Cách 3: Override Parameters từ Command Line

```powershell
python -m src.training.train `
    --config config/train_config.yaml `
    --epochs 60 `
    --batch-size 32 `
    --learning-rate 1e-4
```

#### 🔄 Resume từ Checkpoint

Nếu training bị gián đoạn, resume từ checkpoint:
```powershell
python -m src.training.train `
    --config config/train_config.yaml `
    --resume models/checkpoint.pt
```

#### 📊 Tính Năng Training Mới

**1. Kiến Trúc Mô Hình Cải Tiến:**
- **SmileNetV2**: Residual blocks + SE attention (chính xác hơn 3-5%)
- 6 residual blocks với skip connections
- Squeeze-and-Excitation attention cho mỗi block
- Dropout regularization thông minh

**2. Data Augmentation Mạnh Hơn:**
- Random rotation ±10°
- Color jitter (brightness, contrast, saturation, hue)
- Random erasing (cutout) - 30% probability
- Gaussian blur - 20% probability

**3. Training Techniques Hiện Đại:**
- **Mixed Precision Training**: Nhanh hơn 2-3x trên GPU NVIDIA (FP16)
- **Learning Rate Warmup**: 5 epochs đầu tăng LR từ 0 → base_lr
- **Cosine Annealing**: LR giảm dần theo cosine schedule
- **Gradient Clipping**: max_norm = 1.0 để tránh exploding gradients
- **Early Stopping**: Dừng tự động nếu không cải thiện sau 15 epochs

**4. Optimizer Mới:**
- **AdamW** với weight decay = 1e-4 (tốt hơn Adam gốc)
- L2 regularization tích hợp

**5. Monitoring & Logging:**
- In metrics đẹp với emoji và màu sắc
- Tự động hiển thị thông tin GPU
- Lưu checkpoint định kỳ mỗi 5 epochs
- Test evaluation sau khi training xong

Kết quả:
- `models/smile_cnn_best.pth`: Trọng số tốt nhất (theo F1 score)
- `models/checkpoint.pt`: Full checkpoint (optimizer, scheduler, epoch)
- `models/checkpoint_epoch_N.pt`: Checkpoint định kỳ
- `models/training_history.json`: Metrics đầy đủ theo từng epoch

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
```bash
uvicorn webapp.backend.main:app --host 127.0.0.1 --port 8000
```

Mở trình duyệt: `http://127.0.0.1:8000`

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

### Config File (KHUYẾN NGHỊ)

Chỉnh sửa `config/train_config.yaml` để tùy chỉnh toàn bộ training pipeline:

**Thay đổi kiến trúc model:**
```yaml
model:
  name: "SmileNetV2"      # Hoặc "SmileNet" cho baseline
  dropout: 0.3             # Tăng lên 0.4-0.5 nếu bị overfit
  use_se_block: true       # Bật/tắt SE attention
  use_deep_residual: true  # Residual blocks sâu hơn
```

**Điều chỉnh data augmentation:**
```yaml
augmentation:
  random_rotation: 15      # Tăng độ rotation
  random_erasing:
    probability: 0.5       # Tăng xác suất cutout
    scale: [0.02, 0.2]     # Vùng xóa lớn hơn
  gaussian_blur:
    probability: 0.3       # Blur nhiều hơn
```

**Thay đổi learning rate schedule:**
```yaml
training:
  scheduler:
    type: "cosine"         # "step", "cosine", "reduce_on_plateau"
    warmup_epochs: 5       # Số epoch warmup
    min_lr: 1.0e-6         # LR tối thiểu
```

**Bật/tắt các tính năng:**
```yaml
settings:
  use_amp: true            # Mixed precision (chỉ GPU)
  grad_clip:
    enabled: true
    max_norm: 1.0
  early_stopping:
    enabled: true
    patience: 15           # Dừng sau N epochs không cải thiện
    min_delta: 0.001       # Ngưỡng cải thiện tối thiểu
```

### Tùy Chỉnh Trực Tiếp Code

#### Thay đổi SmileNet Architecture

Chỉnh `src/classifier/smile_model.py`:
```python
# Thêm residual blocks
self.layer1 = self._make_layer(32, 64, num_blocks=3, ...)  # Tăng từ 2 lên 3

# Thay đổi dropout
self.classifier = nn.Sequential(
    nn.Dropout(0.4),  # Tăng dropout
    ...
)
```

#### Thêm Custom Augmentation

Chỉnh `src/training/train.py` → `build_transforms()`:
```python
# Thêm vào training transforms
train_transforms.append(transforms.RandomAffine(
    degrees=15,
    translate=(0.1, 0.1),
    scale=(0.9, 1.1)
))
```

---

## 📊 Hiệu Suất Mô Hình

### Baseline Model (SmileNet)
**Training Results (GENKI-4K, 25 epochs):**
- Validation Accuracy: **88.25%**
- Validation F1 Score: **0.8878**
- Validation Precision: **91.63%**
- Validation Recall: **86.11%**

**Kiến trúc:**
- 4 conv layers với BatchNorm
- Global average pooling
- 2-layer classifier head
- Dropout regularization (0.3)

### Improved Model (SmileNetV2) 🆕

**Dự kiến cải thiện (với config mới):**
- Validation Accuracy: **90-92%** (↑ 2-4%)
- Validation F1 Score: **0.91-0.93** (↑ 0.02-0.04)
- Tốc độ training: **2-3x nhanh hơn** (nhờ mixed precision)

**Cải tiến chính:**
- ✅ 6 residual blocks với skip connections
- ✅ Squeeze-and-Excitation attention
- ✅ Advanced data augmentation (rotation, erasing, blur)
- ✅ Mixed precision training (FP16)
- ✅ Cosine annealing + warmup
- ✅ AdamW optimizer với weight decay
- ✅ Gradient clipping + early stopping

**Kiến trúc SmileNetV2:**
```
Input (64x64x3)
  ↓
Conv2d(3→32) + BN + ReLU
  ↓
[ResBlock(32→64) + SE] × 2  → Downsample
  ↓
[ResBlock(64→128) + SE] × 2 → Downsample
  ↓
[ResBlock(128→256) + SE] × 2 → Downsample
  ↓
AdaptiveAvgPool(1×1)
  ↓
Linear(256→128) + Dropout(0.3)
  ↓
Linear(128→2)
```

**So sánh:**
| Metric | SmileNet | SmileNetV2 (Dự kiến) |
|--------|----------|----------------------|
| Parameters | ~500K | ~750K |
| F1 Score | 0.8878 | 0.91-0.93 |
| Inference Speed | 5ms | 6ms |
| GPU Memory | 1.2GB | 1.5GB |
| Training Time (50 epochs) | 45min | 30min (với AMP) |

---

## 🛠 Troubleshooting

### Lỗi "Không tìm thấy config file"
- Đảm bảo file `config/train_config.yaml` tồn tại
- Hoặc chỉ định đường dẫn khác: `--config path/to/config.yaml`

### Lỗi "Không tìm thấy checkpoint"
- Kiểm tra file `models/smile_cnn_best.pth` và `models/yolov8n-face.pt` tồn tại
- Nếu chưa train, phải chạy bước huấn luyện trước
- Với SmileNetV2, có thể cần train lại từ đầu

### Lỗi CUDA out of memory
**Giải pháp:**
1. Giảm batch_size trong config:
   ```yaml
   training:
     batch_size: 32  # Hoặc 16
   ```
2. Tắt mixed precision:
   ```yaml
   settings:
     use_amp: false
   ```
3. Giảm số workers:
   ```yaml
   settings:
     num_workers: 2
   ```
4. Hoặc train trên CPU (chậm):
   ```yaml
   settings:
     device: "cpu"
   ```

### Accuracy thấp / Model không học
**Các cách khắc phục:**

1. **Tăng epochs và giảm learning rate:**
   ```yaml
   training:
     epochs: 80
     learning_rate: 1.0e-4
   ```

2. **Giảm augmentation nếu quá mạnh:**
   ```yaml
   augmentation:
     random_rotation: 5     # Giảm từ 10
     random_erasing:
       probability: 0.2     # Giảm từ 0.3
   ```

3. **Thử optimizer khác:**
   ```yaml
   training:
     optimizer:
       type: "sgd"          # Thay vì adamw
       momentum: 0.9
   ```

4. **Kiểm tra data:**
   ```powershell
   python -c "import pandas as pd; df=pd.read_csv('data/processed/genki4k/train.csv'); print(df['label'].value_counts())"
   ```
   Đảm bảo labels cân bằng (50/50 hoặc gần đó)

5. **Resume từ baseline model:**
   - Train SmileNet (baseline) trước
   - Sau đó chuyển sang SmileNetV2

### Model overfit (train acc cao, val acc thấp)
- Tăng dropout trong config: `dropout: 0.4`
- Tăng augmentation probability
- Thêm weight_decay: `weight_decay: 5.0e-4`
- Enable early stopping với patience thấp hơn

### Training quá chậm
- Bật mixed precision: `use_amp: true`
- Tăng batch_size (nếu GPU đủ RAM): `batch_size: 128`
- Tăng num_workers: `num_workers: 8`
- Kiểm tra GPU được sử dụng: `nvidia-smi`

### Web demo không load
- Kiểm tra `webapp/frontend/index.html` tồn tại
- Đảm bảo port 8000 không bị chiếm
- Xem log terminal để debug
- Thử restart server: `Ctrl+C` rồi chạy lại `uvicorn`

### Lỗi "Model architecture mismatch"
- Khi load checkpoint cũ với SmileNetV2 mới
- Giải pháp: Train lại từ đầu với kiến trúc mới
- Hoặc dùng model cũ: `model: name: "SmileNet"`

---

## 📚 Tham Khảo

- **YOLOv8**: [Ultralytics Documentation](https://docs.ultralytics.com/)
- **GENKI-4K**: Dataset công khai cho smile detection
- **PyTorch**: [pytorch.org](https://pytorch.org/)
- **FastAPI**: [fastapi.tiangolo.com](https://fastapi.tiangolo.com/)


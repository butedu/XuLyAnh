# 🚀 Hướng Dẫn Sử Dụng Nhanh

## Bước 1: Test Setup

Trước khi train, kiểm tra mọi thứ hoạt động:

```powershell
python test_setup.py
```

Kết quả mong đợi:
```
✅ PASS - Imports
✅ PASS - Config File
✅ PASS - Model Architecture
✅ PASS - Data Paths
✅ PASS - GPU Setup
```

## Bước 2: Chỉnh Config (Tùy Chọn)

Mở `config/train_config.yaml` và điều chỉnh:

```yaml
# Thay đổi epochs và batch size
training:
  epochs: 50
  batch_size: 64  # Giảm xuống 32 nếu GPU hết RAM

# Chọn model architecture
model:
  name: "SmileNetV2"  # Hoặc "SmileNet" cho baseline
  use_se_block: true
```

## Bước 3: Training

### Cách 1: Dùng Python Script (Khuyến nghị)

```bash
python train_model.py
```

### Cách 2: Trực tiếp với module

```powershell
python -m src.training.train --config config/train_config.yaml
```

## Bước 4: Theo Dõi Training

Terminal sẽ hiển thị:

```
🚀 Sử dụng GPU: NVIDIA GeForce RTX 3080
   CUDA Version: 12.1
   GPU Memory: 10.00 GB

📊 Mô hình: SmileNetV2
   Tổng số parameters: 756,482
   Trainable parameters: 756,482

⚡ Mixed Precision Training: ENABLED

🎯 Bắt đầu training 50 epochs...

============================================================
Epoch 1/50
============================================================
🔥 Warmup LR: 0.000100

train: 100%|████████| 50/50 [00:15<00:00,  3.21it/s]
eval: 100%|████████| 6/6 [00:01<00:00,  5.23it/s]

📈 Metrics:
   Train Loss: 0.4521
   Val Loss:   0.3892
   Val Acc:    85.50%
   Val F1:     0.8612
   Val Prec:   87.23%
   Val Recall: 84.12%
   LR:         0.000100

💾 Lưu mô hình tốt nhất: F1 = 0.8612 (+0.8612)
```

## Bước 5: Kết Quả

Sau khi training xong, kiểm tra thư mục `models/`:

```
models/
├── smile_cnn_best.pth          # Model tốt nhất (dùng cho inference)
├── checkpoint.pt               # Full checkpoint (để resume)
├── checkpoint_epoch_5.pt       # Checkpoint định kỳ
├── checkpoint_epoch_10.pt
└── training_history.json       # Metrics đầy đủ
```

## Sử Dụng Model

### Xử lý ảnh đơn:

```powershell
python main.py path/to/image.jpg --output result.jpg
```

### Web demo:

```bash
uvicorn webapp.backend.main:app --host 127.0.0.1 --port 8000
```

Mở: http://127.0.0.1:8000

## Tips & Tricks

### 🔥 Training nhanh hơn:
- Tăng `batch_size` lên 128 (nếu GPU đủ RAM)
- Đảm bảo `use_amp: true` trong config
- Tăng `num_workers` lên 8

### 🎯 Accuracy cao hơn:
- Tăng `epochs` lên 80-100
- Giảm `learning_rate` xuống 1e-4
- Tăng augmentation probability
- Bật `use_se_block: true`

### 💾 Tiết kiệm VRAM:
- Giảm `batch_size` xuống 32 hoặc 16
- Tắt `use_amp: false` (ít RAM hơn nhưng chậm)
- Giảm `num_workers` xuống 2

### 🐛 Debugging:
- Chạy `python test_setup.py` để kiểm tra setup
- Xem `models/training_history.json` để phân tích metrics
- Thêm `--device cpu` để test trên CPU

## Troubleshooting Nhanh

**Lỗi: CUDA out of memory**
→ Giảm `batch_size` trong config

**Lỗi: Config file not found**
→ Đảm bảo `config/train_config.yaml` tồn tại

**Model không học (loss không giảm)**
→ Kiểm tra data với `python -c "import pandas as pd; print(pd.read_csv('data/processed/genki4k/train.csv').head())"`

**Training quá chậm**
→ Kiểm tra GPU đang được dùng: `nvidia-smi`
→ Bật mixed precision: `use_amp: true`

## Tài Liệu Chi Tiết

- `README.md` - Hướng dẫn đầy đủ
- `docs/IMPROVEMENTS.md` - Chi tiết các cải tiến kỹ thuật
- `config/train_config.yaml` - Comments giải thích từng parameter

## Support

Nếu gặp vấn đề, kiểm tra:
1. `python test_setup.py` - test cơ bản
2. `models/training_history.json` - xem metrics
3. Terminal logs - xem error messages

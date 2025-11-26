# HƯỚNG DẪN SỬ DỤNG CHI TIẾT

## Mục lục
1. [Giới thiệu](#giới-thiệu)
2. [Cài đặt](#cài-đặt)
3. [Chuẩn bị dữ liệu](#chuẩn-bị-dữ-liệu)
4. [Huấn luyện mô hình](#huấn-luyện-mô-hình)
5. [Sử dụng mô hình](#sử-dụng-mô-hình)
6. [Demo web](#demo-web)
7. [Giải thích code](#giải-thích-code)

## Giới thiệu

Dự án này giải quyết bài toán: **Đếm số người cười trong ảnh nhóm**

### Pipeline hoạt động:
```
Ảnh đầu vào
    ↓
Phát hiện khuôn mặt (Haar Cascade hoặc DNN)
    ↓
Cắt từng khuôn mặt
    ↓
Phân loại cười/không cười (CNN)
    ↓
Thống kê + Vẽ kết quả
```

### Cấu trúc dự án:
- `src/models/`: Chứa mô hình CNN và face detector
- `src/training/`: Code huấn luyện
- `src/inference/`: Code suy luận
- `src/data/`: Script chuẩn bị dữ liệu
- `webapp/`: Ứng dụng web demo
- `models/`: Lưu trọng số đã huấn luyện

## Cài đặt

### Bước 1: Cài Python
Cần Python >= 3.10

### Bước 2: Tạo môi trường ảo
```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

### Bước 3: Cài thư viện
```powershell
pip install -r requirements.txt
```

**Các thư viện chính:**
- `torch`, `torchvision`: Framework deep learning
- `opencv-python`: Xử lý ảnh, phát hiện khuôn mặt
- `numpy`, `pandas`: Xử lý dữ liệu
- `scikit-learn`: Chia tập train/val/test, tính metrics
- `fastapi`, `uvicorn`: Web framework
- `pillow`: Đọc/xử lý ảnh
- `tqdm`: Progress bar

## Chuẩn bị dữ liệu

### Dataset 1: GENKI-4K

**Tải về:**
1. Truy cập: https://inc.ucsd.edu/mplab/398/
2. Tải file GENKI-4K.zip
3. Giải nén vào `data/raw/genki4k/`

**Cấu trúc sau khi giải nén:**
```
data/raw/genki4k/
├── files/          # 4000 ảnh khuôn mặt
└── labels.txt      # Nhãn: 1=cười, -1=không cười
```

**Chuyển đổi sang CSV:**
```powershell
python -m src.data.prepare_genki --raw-dir data/raw/genki4k --output-dir data/processed/genki4k --create-splits
```

**Kết quả:**
```
data/processed/genki4k/
├── annotations.csv  # Toàn bộ dữ liệu
├── train.csv       # 80% cho training
├── val.csv         # 10% cho validation
└── test.csv        # 10% cho testing
```

### Dataset 2: RAF-DB (tùy chọn)

**Tải về:**
1. Đăng ký tại: http://www.whdeng.cn/RAF/model1.html
2. Tải RAF-DB dataset
3. Giải nén vào `data/raw/rafdb/`

**Chuyển đổi:**
```powershell
python -m src.data.prepare_rafdb --raw-dir data/raw/rafdb --output-dir data/processed/rafdb
```

**Lưu ý:** RAF-DB có 7 biểu cảm, script sẽ gộp "happiness" thành "cười", còn lại là "không cười".

### Tổ chức ảnh để huấn luyện

**Cách 1: Sử dụng GENKI-4K**
```powershell
# Copy ảnh vào thư mục chung
mkdir data/images
cp -r data/raw/genki4k/files/* data/images/
```

**Cách 2: Kết hợp nhiều dataset**
Gộp các file CSV và đảm bảo cột `filepath` trỏ đúng vị trí ảnh.

## Huấn luyện mô hình

### Mô hình CNN

**Kiến trúc:**
```
Input (64x64x3)
    ↓
Conv2D(32) → BatchNorm → ReLU → MaxPool  # 32x32
    ↓
Conv2D(64) → BatchNorm → ReLU → MaxPool  # 16x16
    ↓
Conv2D(128) → BatchNorm → ReLU → MaxPool # 8x8
    ↓
Conv2D(256) → BatchNorm → ReLU → AdaptiveAvgPool # 2x2
    ↓
Flatten → Dropout(0.35) → Linear(256) → ReLU
    ↓
Dropout(0.35) → Linear(2) # Output: [không cười, cười]
```

**Tính năng:**
- BatchNorm: Ổn định training
- Dropout: Tránh overfitting
- AdaptiveAvgPool: Linh hoạt với kích thước đầu vào

### Lệnh huấn luyện

**Cơ bản:**
```powershell
python -m src.training.train_smile_classifier `
    --image-root data/images `
    --split-dir data/processed/genki4k `
    --output-dir models
```

**Với tùy chỉnh:**
```powershell
python -m src.training.train_smile_classifier `
    --image-root data/images `
    --split-dir data/processed/genki4k `
    --output-dir models `
    --epochs 30 `
    --batch-size 64 `
    --learning-rate 0.001 `
    --step-size 10 `
    --gamma 0.5 `
    --device cuda
```

**Tham số:**
- `--epochs`: Số epoch huấn luyện (mặc định: 25)
- `--batch-size`: Kích thước batch (mặc định: 64)
- `--learning-rate`: Learning rate (mặc định: 0.001)
- `--step-size`: Giảm LR sau mỗi N epochs (mặc định: 10)
- `--gamma`: Hệ số giảm LR (mặc định: 0.5)
- `--device`: cpu hoặc cuda

### Kết quả huấn luyện

```
models/
├── smile_cnn_best.pth      # Trọng số mô hình tốt nhất
├── checkpoint.pt           # Checkpoint đầy đủ (optimizer, scheduler)
└── training_history.json   # Lịch sử loss và metrics
```

## Sử dụng mô hình

### Phương pháp 1: Script Python

**File:** `vi_du_su_dung.py`

```python
from src.inference.smile_detector import BoNhanDienCuoi, CauHinhBoNhanDienCuoi
import cv2

# Khởi tạo
cau_hinh = CauHinhBoNhanDienCuoi(
    face_backend="haar",
    model_weights="models/smile_cnn_best.pth"
)
bo_phat_hien = BoNhanDienCuoi(cau_hinh)

# Đọc ảnh
anh = cv2.imread("path/to/image.jpg")

# Phân tích
ket_qua = bo_phat_hien.phan_tich(anh)

# In kết quả
print(f"Tổng khuôn mặt: {ket_qua['total_faces']}")
print(f"Số người cười: {ket_qua['smiling_faces']}")

# Vẽ chú thích
anh_ket_qua = bo_phat_hien.chu_thich_anh(anh, ket_qua['detections'])
cv2.imwrite("result.jpg", anh_ket_qua)
```

**Chạy ví dụ:**
```powershell
python vi_du_su_dung.py
```

### Phương pháp 2: Web Demo

**Bước 1: Khởi động server**
```powershell
uvicorn webapp.backend.main:app --reload --port 8000
```

**Bước 2: Mở trình duyệt**
```
http://localhost:8000
```

**Bước 3: Tải ảnh lên**
- Kéo thả hoặc chọn file ảnh
- Bấm "Phân tích ảnh"
- Xem kết quả với ảnh đã được đánh dấu

## Giải thích code

### 1. Mô hình CNN (`src/models/smile_cnn.py`)

**Class `SmileCNN`:**
```python
class SmileCNN(nn.Module):
    def __init__(self, input_channels=3, dropout=0.35):
        # Khởi tạo các lớp mạng
        
    def forward(self, x):
        # Lan truyền xuôi: ảnh -> logits
```

**Hàm tiện ích:**
- `xay_dung_mo_hinh()`: Tạo mô hình mới
- `tai_mo_hinh()`: Tải từ checkpoint

### 2. Face Detector (`src/models/face_detector.py`)

**Hai phương pháp:**

**Haar Cascade:**
- Ưu: Nhanh, không cần GPU
- Nhược: Độ chính xác thấp với góc nghiêng

**DNN (ResNet-10):**
- Ưu: Chính xác cao
- Nhược: Chậm hơn, cần model riêng

**Sử dụng:**
```python
from src.models.face_detector import BoNhanDienKhuonMat, CauHinhBoNhanDienKhuonMat

cau_hinh = CauHinhBoNhanDienKhuonMat(
    backend="haar",  # hoặc "dnn"
    min_confidence=0.5
)
detector = BoNhanDienKhuonMat(cau_hinh)
boxes = detector.phat_hien(anh_bgr)
```

### 3. Pipeline suy luận (`src/inference/smile_detector.py`)

**Class `BoNhanDienCuoi`:**

**Phương thức chính:**
1. `phat_hien_khuon_mat()`: Tìm khuôn mặt
2. `du_doan_cuoi()`: Phân loại từng khuôn mặt
3. `phan_tich()`: Pipeline đầy đủ
4. `chu_thich_anh()`: Vẽ kết quả

**Quy trình:**
```python
# 1. Phát hiện khuôn mặt
boxes = self.phat_hien_khuon_mat(anh)

# 2. Với mỗi khuôn mặt
for x, y, w, h, conf in boxes:
    khuon_mat = anh[y:y+h, x:x+w]
    
    # 3. Phân loại
    nhan, xac_suat = self.du_doan_cuoi(khuon_mat)
    
    # 4. Lưu kết quả
    if nhan == 1:
        so_nguoi_cuoi += 1
```

### 4. Web API (`webapp/backend/main.py`)

**Endpoint:**
- `GET /`: Trang chủ
- `POST /api/detect`: Upload ảnh, nhận kết quả JSON

**Flow:**
```
Client upload ảnh
    ↓
FastAPI nhận file
    ↓
DichVuNhanDienCuoi xử lý
    ↓
Trả JSON: {summary, annotated_image_base64}
    ↓
Client hiển thị kết quả
```

## Tips & Tricks

### 1. Cải thiện độ chính xác

**Data Augmentation:**
```python
transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
])
```

**Tăng số epoch:**
```powershell
--epochs 50
```

**Thử backbone mạnh hơn:**
Thay CNN nhẹ bằng ResNet-18, MobileNetV2

### 2. Tăng tốc độ

**Giảm batch size nếu thiếu RAM:**
```powershell
--batch-size 32
```

**Sử dụng Haar thay vì DNN:**
```python
face_backend="haar"
```

**Giảm kích thước ảnh:**
```python
MODEL_IMAGE_SIZE = (48, 48)  # thay vì (64, 64)
```

### 3. Xử lý lỗi thường gặp

**Lỗi: "Module not found"**
```powershell
# Đảm bảo đang ở thư mục gốc
cd d:\Dulieuhoc\Dulieuhoc\XuLyAnh\XuLyAnh
python -m src.training.train_smile_classifier ...
```

**Lỗi: "CUDA out of memory"**
```powershell
--batch-size 16 --device cpu
```

**Lỗi: "File not found: labels.txt"**
Kiểm tra cấu trúc thư mục dataset đúng chưa

## Kết luận

Dự án cung cấp pipeline hoàn chỉnh từ chuẩn bị dữ liệu → huấn luyện → suy luận → demo web. Tất cả code đã được dịch sang tiếng Việt với chú thích chi tiết để dễ hiểu và tùy chỉnh.

**Liên hệ và góp ý:**
- Mở issue trên GitHub nếu gặp vấn đề
- Đóng góp code qua Pull Request

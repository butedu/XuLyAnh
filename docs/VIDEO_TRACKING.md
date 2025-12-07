# 🎯 Face Tracking trong Video - Hướng Dẫn Chi Tiết

## Vấn Đề với Video Processing Cũ

### ❌ Vấn đề:
1. **Mỗi frame xử lý độc lập** → Không biết ai là ai giữa các frames
2. **Cùng 1 người nhưng bị đếm nhiều lần** → Thống kê sai
3. **Xử lý tất cả frames** → Chậm, tốn GPU
4. **Khi chuyển cảnh** → Mất tracking, ID bị reset không đúng

### ✅ Giải pháp: Face Tracking System

## Kiến Trúc Hệ Thống

```
Video Frame → [Detector] → Faces → [Tracker] → Tracked Faces (với ID)
                  ↓                      ↓
            Chỉ chạy mỗi N frames   Chạy mọi frame (nhanh)
                  ↓                      ↓
            Smile Classification    Update tracking state
                  ↓                      ↓
            Smile Probability      Gán ID + Tính thống kê
```

## Thành Phần Chính

### 1. SimpleFaceTracker

**Chức năng:**
- Gán ID cố định cho mỗi khuôn mặt
- Track khuôn mặt xuyên suốt video
- Tính toán thống kê theo từng người (không phải từng frame)

**Thuật toán:**
1. **IoU Matching**: Match detection mới với track cũ bằng Intersection over Union
2. **Motion Prediction**: Dự đoán vị trí bbox ở frame tiếp theo dựa trên velocity
3. **Distance Threshold**: Giới hạn khoảng cách để tránh matching sai
4. **Age-based Filtering**: Chỉ confirm track sau N detections (tránh false positive)

### 2. Scene Change Detection

**Vấn đề:**
- Khi video chuyển cảnh, người trong cảnh cũ và cảnh mới khác nhau
- Nếu không reset tracker → ID sẽ bị gán sai

**Giải pháp:**
```python
def detect_scene_change(prev_frame, curr_frame):
    # Tính histogram difference
    # Nếu difference > threshold → Scene change
    # → Reset tracker
```

**Khi nào reset:**
- Histogram distance > 0.35 (default)
- Tự động phát hiện cuts, fades, transitions

### 3. Sparse Frame Processing

**Ý tưởng:**
- Detector (YOLO + CNN) nặng → Chạy mỗi N frames
- Tracker nhẹ → Chạy mọi frame

**Ví dụ:**
```
Frame 1: [Detect + Classify] → Update Tracker
Frame 2: [Skip detection]    → Tracker dùng prediction
Frame 3: [Skip detection]    → Tracker dùng prediction
Frame 4: [Detect + Classify] → Update Tracker với data mới
...
```

**Lợi ích:**
- ⚡ Nhanh hơn 3-5x (nếu process_every=3)
- 🎯 Vẫn track chính xác (tracker interpolate giữa các detections)
- 💾 Tiết kiệm GPU memory

## Tracking Workflow Chi Tiết

### Bước 1: Initialization
```python
tracker = SimpleFaceTracker(
    iou_threshold=0.3,      # IoU tối thiểu để match
    max_age=30,             # Số frames tối đa không detect trước khi xóa
    min_hits=3,             # Số detections tối thiểu để confirm
    distance_threshold=150  # Khoảng cách tối đa (pixels)
)
```

### Bước 2: Processing Loop

**Mỗi frame:**
1. Check scene change
   - Nếu có → Reset tracker
   
2. Detection (nếu frame_idx % process_every == 0)
   - YOLO detect faces
   - CNN classify smiles
   - Lưu detections
   
3. Update tracker với detections
   - Match detections với tracks hiện tại
   - Update matched tracks
   - Tạo tracks mới cho unmatched detections
   - Xóa tracks cũ (quá lâu không detect)
   
4. Vẽ tracked faces lên frame
   - ID + Smile probability
   - Smile ratio (% frames cười)
   
5. Thu thập statistics

### Bước 3: Matching Algorithm

```python
def match_detections_to_tracks(detections, tracks):
    # 1. Dự đoán vị trí tracks ở frame hiện tại
    for track in tracks:
        predicted_bbox = track.predict_next_bbox()
    
    # 2. Tính cost matrix (1 - IoU)
    cost_matrix = np.zeros((n_detections, n_tracks))
    for i, detection in enumerate(detections):
        for j, track in enumerate(tracks):
            iou = compute_iou(detection.bbox, track.predicted_bbox)
            distance = compute_distance(detection.bbox, track.predicted_bbox)
            
            if distance > threshold:
                cost[i,j] = infinity  # Quá xa, không match
            else:
                cost[i,j] = 1 - iou
    
    # 3. Greedy matching
    # Sắp xếp theo cost tăng dần
    # Match từng cặp (detection, track) nếu:
    #   - Chưa được match
    #   - IoU > threshold
    
    return matched_pairs, unmatched_detections, unmatched_tracks
```

### Bước 4: Track Management

**Confirmed Tracks:**
- Track chỉ được coi là "confirmed" sau `min_hits` detections
- Tránh false positive từ detection noise

**Track Removal:**
- Nếu track không được update (detect) trong `max_age` frames → Xóa
- Tránh giữ tracks của người đã ra khỏi khung hình

## Thống Kê Chi Tiết

### Per-Person Statistics

Mỗi track (người) có:

```python
track_id: int                    # ID duy nhất
total_frames: int                # Tổng số frames xuất hiện
smile_frames: int                # Số frames cười
smile_ratio: float               # % frames cười
duration: float                  # Thời lượng xuất hiện (giây)
smile_duration: float            # Thời lượng cười (giây)
```

### Global Statistics

```python
total_people: int                # Tổng số người trong video
people_smiling: int              # Số người cười (smile_ratio ≥ 30%)
```

## Sử Dụng

### Cơ Bản

```bash
python video_demo_tracking.py input.mp4 --output output.mp4
```

### Tùy Chỉnh

```bash
# Process mỗi 5 frames (nhanh hơn)
python video_demo_tracking.py input.mp4 --process-every 5

# Điều chỉnh scene change sensitivity
python video_demo_tracking.py input.mp4 --scene-threshold 0.4

# Dùng CPU
python video_demo_tracking.py input.mp4 --device cpu

# Custom model
python video_demo_tracking.py input.mp4 --weights models/my_model.pth
```

## Kết Quả

### Output Video

Mỗi khuôn mặt hiển thị:
- **Track ID**: Số ID cố định
- **Probability**: Xác suất cười ở frame hiện tại
- **Smile Ratio**: % tổng thời gian cười
- **Bounding Box**: Xanh (cười) hoặc Đỏ (không cười)

### Console Output

```
📊 Final Statistics
============================================================
Total People Tracked: 3
People Smiling: 2

Per-person breakdown:
  ID 1:
    - Duration: 12.3s (370 frames)
    - Smiling: 8.5s (255 frames, 68.9%)
    - Status: 😊 Smiling

  ID 2:
    - Duration: 15.7s (471 frames)
    - Smiling: 3.2s (96 frames, 20.4%)
    - Status: 😐 Neutral

  ID 3:
    - Duration: 10.1s (303 frames)
    - Smiling: 9.8s (294 frames, 97.0%)
    - Status: 😊 Smiling
```

## Tối Ưu Performance

### Giảm Processing Time

1. **Tăng process_every**
   ```bash
   --process-every 5  # Chỉ detect mỗi 5 frames (5x nhanh hơn)
   ```

2. **Giảm resolution**
   - Resize video trước khi process
   - Hoặc chỉnh trong code

3. **Dùng batch processing**
   - Process nhiều frames cùng lúc (nếu GPU đủ mạnh)

### Tăng Accuracy

1. **Giảm process_every**
   ```bash
   --process-every 1  # Detect mọi frame (chậm nhưng chính xác)
   ```

2. **Tăng min_hits**
   ```python
   SimpleFaceTracker(min_hits=5)  # Require 5 detections để confirm
   ```

3. **Giảm max_age**
   ```python
   SimpleFaceTracker(max_age=15)  # Xóa track nhanh hơn nếu mất
   ```

## So Sánh

| Feature | video_demo.py (Cũ) | video_demo_tracking.py (Mới) |
|---------|-------------------|------------------------------|
| **Tracking** | ❌ Không | ✅ Có (với ID cố định) |
| **Thống kê** | ❌ Theo frame | ✅ Theo người |
| **Scene change** | ❌ Không xử lý | ✅ Tự động detect + reset |
| **Performance** | Chậm (process mọi frame) | Nhanh 3-5x (sparse processing) |
| **Độ chính xác** | Trung bình | Cao hơn |
| **Use case** | Video ngắn, đơn giản | Video dài, nhiều người, chuyển cảnh |

## Troubleshooting

### ID bị nhảy lung tung
→ Giảm `iou_threshold` hoặc tăng `distance_threshold`

### Tracking bị mất khi người di chuyển nhanh
→ Giảm `process_every` (detect thường xuyên hơn)

### Scene change không được detect
→ Giảm `scene_threshold` (nhạy hơn)

### False positive tracking (detect người không có thật)
→ Tăng `min_hits` (require nhiều detections hơn)

### Video bị lag
→ Tăng `process_every` hoặc dùng CPU cho detector

## Next Steps

- [ ] Thêm Re-ID (face recognition) để track xuyên scene
- [ ] Deep SORT với appearance features
- [ ] Multi-camera tracking
- [ ] Real-time tracking cho webcam

"""
Face tracking module với Deep SORT hoặc ByteTrack để tracking khuôn mặt trong video.
Gán ID cố định cho mỗi người và tính toán thống kê chính xác.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, deque


@dataclass
class FaceTrack:
    """Thông tin tracking của một khuôn mặt."""
    track_id: int
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    smile_probability: float
    is_smiling: bool
    feature_embedding: Optional[np.ndarray] = None
    age: int = 0  # Số frames đã track
    frames_since_detection: int = 0  # Frames kể từ detection cuối
    
    # Lịch sử
    smile_history: List[bool] = field(default_factory=list)
    bbox_history: deque = field(default_factory=lambda: deque(maxlen=30))  # 30 frames lịch sử
    
    def update(self, bbox: Tuple[int, int, int, int], smile_prob: float, is_smiling: bool):
        """Cập nhật track với detection mới."""
        self.bbox = bbox
        self.smile_probability = smile_prob
        self.is_smiling = is_smiling
        self.smile_history.append(is_smiling)
        self.bbox_history.append(bbox)
        self.age += 1
        self.frames_since_detection = 0
    
    def predict_next_bbox(self) -> Tuple[int, int, int, int]:
        """Dự đoán bbox ở frame tiếp theo dựa trên velocity."""
        if len(self.bbox_history) < 2:
            return self.bbox
        
        # Tính velocity từ 2 bbox gần nhất
        prev_bbox = self.bbox_history[-2]
        curr_bbox = self.bbox_history[-1]
        
        dx = curr_bbox[0] - prev_bbox[0]
        dy = curr_bbox[1] - prev_bbox[1]
        
        # Dự đoán bbox tiếp theo
        x = curr_bbox[0] + dx
        y = curr_bbox[1] + dy
        w = curr_bbox[2]
        h = curr_bbox[3]
        
        return (int(x), int(y), int(w), int(h))
    
    def get_smile_ratio(self) -> float:
        """Tính tỷ lệ cười trong tất cả frames đã track."""
        if not self.smile_history:
            return 0.0
        return sum(self.smile_history) / len(self.smile_history)
    
    def get_total_smile_frames(self) -> int:
        """Số frames người này cười."""
        return sum(self.smile_history)
    
    @property
    def smile_count(self) -> int:
        """Số frames cười (alias cho get_total_smile_frames)."""
        return sum(self.smile_history)
    
    @property
    def total_detections(self) -> int:
        """Tổng số detections (frames)."""
        return len(self.smile_history)


def compute_iou(bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
    """Tính IoU (Intersection over Union) giữa 2 bounding boxes."""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    # Tọa độ góc
    x1_max = x1 + w1
    y1_max = y1 + h1
    x2_max = x2 + w2
    y2_max = y2 + h2
    
    # Diện tích giao
    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(x1_max, x2_max)
    inter_y2 = min(y1_max, y2_max)
    
    if inter_x2 < inter_x1 or inter_y2 < inter_y1:
        return 0.0
    
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    
    # Diện tích hợp
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def compute_bbox_distance(bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
    """Tính khoảng cách giữa center của 2 bboxes."""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    center1_x = x1 + w1 / 2
    center1_y = y1 + h1 / 2
    center2_x = x2 + w2 / 2
    center2_y = y2 + h2 / 2
    
    return np.sqrt((center1_x - center2_x)**2 + (center1_y - center2_y)**2)


class SimpleFaceTracker:
    """
    Simple face tracker sử dụng IoU và motion prediction.
    Không cần deep learning models, chạy nhanh.
    """
    
    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_age: int = 30,  # Số frames tối đa không detect được trước khi xóa track
        min_hits: int = 3,  # Số detections tối thiểu để xác nhận track
        distance_threshold: float = 100,  # Khoảng cách tối đa (pixels)
    ):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.distance_threshold = distance_threshold
        
        self.tracks: Dict[int, FaceTrack] = {}
        self.next_id = 1
        self.frame_count = 0
    
    def update(
        self,
        detections: List[Dict[str, any]]
    ) -> List[FaceTrack]:
        """
        Cập nhật tracker với detections mới.
        
        Args:
            detections: List of dict với keys: 'bbox', 'smile_probability', 'is_smiling'
        
        Returns:
            List of confirmed FaceTracks
        """
        self.frame_count += 1
        
        # Dự đoán vị trí tracks hiện tại
        for track in self.tracks.values():
            track.frames_since_detection += 1
        
        # Matching detections với tracks
        if detections and self.tracks:
            matched_pairs, unmatched_detections, unmatched_tracks = self._match_detections_to_tracks(detections)
            
            # Cập nhật matched tracks
            for det_idx, track_id in matched_pairs:
                detection = detections[det_idx]
                self.tracks[track_id].update(
                    detection['bbox'],
                    detection['smile_probability'],
                    detection['is_smiling']
                )
            
            # Tạo tracks mới cho unmatched detections
            for det_idx in unmatched_detections:
                detection = detections[det_idx]
                new_track = FaceTrack(
                    track_id=self.next_id,
                    bbox=detection['bbox'],
                    smile_probability=detection['smile_probability'],
                    is_smiling=detection['is_smiling']
                )
                new_track.update(
                    detection['bbox'],
                    detection['smile_probability'],
                    detection['is_smiling']
                )
                self.tracks[self.next_id] = new_track
                self.next_id += 1
        
        elif detections:
            # Không có tracks hiện tại, tạo mới tất cả
            for detection in detections:
                new_track = FaceTrack(
                    track_id=self.next_id,
                    bbox=detection['bbox'],
                    smile_probability=detection['smile_probability'],
                    is_smiling=detection['is_smiling']
                )
                new_track.update(
                    detection['bbox'],
                    detection['smile_probability'],
                    detection['is_smiling']
                )
                self.tracks[self.next_id] = new_track
                self.next_id += 1
        
        # Xóa tracks cũ (quá lâu không detect)
        tracks_to_remove = []
        for track_id, track in self.tracks.items():
            if track.frames_since_detection > self.max_age:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
        
        # Trả về confirmed tracks (đã track đủ số frames)
        confirmed_tracks = [
            track for track in self.tracks.values()
            if track.age >= self.min_hits
        ]
        
        return confirmed_tracks
    
    def _match_detections_to_tracks(
        self,
        detections: List[Dict[str, any]]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Match detections với tracks bằng Hungarian algorithm đơn giản.
        
        Returns:
            matched_pairs: List of (detection_idx, track_id)
            unmatched_detections: List of detection indices
            unmatched_tracks: List of track IDs
        """
        if not detections or not self.tracks:
            return [], list(range(len(detections))), list(self.tracks.keys())
        
        # Tính cost matrix (1 - IoU)
        track_ids = list(self.tracks.keys())
        cost_matrix = np.zeros((len(detections), len(track_ids)))
        
        for i, detection in enumerate(detections):
            det_bbox = detection['bbox']
            for j, track_id in enumerate(track_ids):
                track = self.tracks[track_id]
                predicted_bbox = track.predict_next_bbox()
                
                # Tính IoU
                iou = compute_iou(det_bbox, predicted_bbox)
                
                # Tính distance
                distance = compute_bbox_distance(det_bbox, predicted_bbox)
                
                # Cost = 1 - IoU, nhưng nếu distance quá xa thì set cost = vô cực
                if distance > self.distance_threshold:
                    cost_matrix[i, j] = 1e6
                else:
                    cost_matrix[i, j] = 1 - iou
        
        # Greedy matching (đơn giản hóa thay vì Hungarian)
        matched_pairs = []
        unmatched_detections = set(range(len(detections)))
        unmatched_tracks = set(track_ids)
        
        # Sort by cost
        matches = []
        for i in range(len(detections)):
            for j in range(len(track_ids)):
                if cost_matrix[i, j] < 1 - self.iou_threshold:  # IoU > threshold
                    matches.append((cost_matrix[i, j], i, j))
        
        matches.sort()  # Sort by cost ascending
        
        for cost, det_idx, track_idx in matches:
            if det_idx in unmatched_detections and track_ids[track_idx] in unmatched_tracks:
                matched_pairs.append((det_idx, track_ids[track_idx]))
                unmatched_detections.remove(det_idx)
                unmatched_tracks.remove(track_ids[track_idx])
        
        return matched_pairs, list(unmatched_detections), list(unmatched_tracks)
    
    def get_statistics(self) -> Dict[str, any]:
        """Lấy thống kê tổng quan."""
        confirmed_tracks = [t for t in self.tracks.values() if t.age >= self.min_hits]
        
        if not confirmed_tracks:
            return {
                'total_people': 0,
                'people_smiling': 0,
                'tracks': []
            }
        
        track_stats = []
        people_smiling = 0
        
        for track in confirmed_tracks:
            smile_ratio = track.get_smile_ratio()
            total_frames = track.age
            smile_frames = track.get_total_smile_frames()
            
            # Coi là "người cười" nếu cười ít nhất 30% thời gian
            is_person_smiling = smile_ratio >= 0.3
            if is_person_smiling:
                people_smiling += 1
            
            track_stats.append({
                'track_id': track.track_id,
                'total_frames': total_frames,
                'smile_frames': smile_frames,
                'smile_ratio': smile_ratio,
                'is_smiling': is_person_smiling,
                'current_bbox': track.bbox
            })
        
        return {
            'total_people': len(confirmed_tracks),
            'people_smiling': people_smiling,
            'tracks': track_stats
        }
    
    def reset(self):
        """Reset tracker (dùng khi chuyển cảnh)."""
        self.tracks.clear()
        self.next_id = 1
        self.frame_count = 0
    
    def get_all_tracks(self) -> List[FaceTrack]:
        """Lấy tất cả tracks (bao gồm cả chưa confirmed)."""
        return list(self.tracks.values())


def detect_scene_change(
    prev_frame: np.ndarray,
    curr_frame: np.ndarray,
    threshold: float = 30.0
) -> bool:
    """
    Phát hiện scene change bằng histogram difference.
    
    Args:
        prev_frame: Frame trước (BGR)
        curr_frame: Frame hiện tại (BGR)
        threshold: Ngưỡng để coi là scene change
    
    Returns:
        True nếu phát hiện scene change
    """
    if prev_frame is None or curr_frame is None:
        return False
    
    # Resize về kích thước nhỏ để tính nhanh
    import cv2
    prev_small = cv2.resize(prev_frame, (64, 64))
    curr_small = cv2.resize(curr_frame, (64, 64))
    
    # Tính histogram
    prev_hist = cv2.calcHist([prev_small], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    curr_hist = cv2.calcHist([curr_small], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    
    # Normalize
    cv2.normalize(prev_hist, prev_hist)
    cv2.normalize(curr_hist, curr_hist)
    
    # Tính distance
    distance = cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_BHATTACHARYYA)
    
    return distance > threshold

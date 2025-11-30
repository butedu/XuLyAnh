import os
import sys
from typing import List, Tuple

from PIL import Image

try:
    from facenet_pytorch import MTCNN
except Exception:
    MTCNN = None


class MTCNNFaceDetector:
    """Lightweight wrapper around facenet-pytorch MTCNN.

    Returns boxes in (x1, y1, x2, y2) format and crops as PIL images.
    If facenet-pytorch is not installed, raises ImportError at init.
    """

    def __init__(self, device: str = 'cpu', keep_all: bool = True, min_face_size: int = 20):
        if MTCNN is None:
            raise ImportError("facenet-pytorch is required for MTCNNFaceDetector. Install via `pip install facenet-pytorch`")
        self.device = device
        self.mtcnn = MTCNN(keep_all=keep_all, device=device, min_face_size=min_face_size)

    def detect(self, pil_image: Image.Image) -> List[Tuple[Tuple[int,int,int,int], float]]:
        """Detect faces and return list of (bbox, score).

        bbox is (x1,y1,x2,y2), coordinates are integers.
        score is float (confidence from MTCNN) or 0.0 if unavailable.
        """
        boxes, probs = self.mtcnn.detect(pil_image)
        results = []
        if boxes is None:
            return results
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = [int(x) for x in box]
            score = float(probs[i]) if probs is not None else 0.0
            results.append(((x1, y1, x2, y2), score))
        return results

    def crop(self, pil_image: Image.Image, bbox: Tuple[int,int,int,int], margin: float = 0.2) -> Image.Image:
        x1,y1,x2,y2 = bbox
        w = x2 - x1
        h = y2 - y1
        mx = int(w * margin)
        my = int(h * margin)
        x1c = max(0, x1 - mx)
        y1c = max(0, y1 - my)
        x2c = min(pil_image.width, x2 + mx)
        y2c = min(pil_image.height, y2 + my)
        return pil_image.crop((x1c, y1c, x2c, y2c))

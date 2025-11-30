import os
import sys
from typing import Any, Dict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.inference.advanced_smile_detector import AdvancedSmileDetector


class DichVuNhanDienCuoiNangCao:
    """Advanced service wrapper — uses MTCNN-based detector + existing classifier.

    This class can be used by a separate FastAPI app to avoid modifying original service code.
    """

    def __init__(self, model_path: str = None, device: str = None):
        self.detector = AdvancedSmileDetector(model_path=model_path, device=device)

    def phan_tich_tu_file(self, file_path: str) -> Dict[str, Any]:
        results = self.detector.analyze_file(file_path)
        return {'results': results}

import os
import sys
from typing import List, Dict, Tuple

from PIL import Image
import torch
from torchvision import transforms

# Ensure project root is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from src.models.smile_cnn import SmileCNN
except Exception:
    SmileCNN = None

from .ensemble_detector import EnsembleFaceDetector


class AdvancedSmileDetector:
    """Detect faces with MTCNN then classify smile using existing SmileCNN model.

    This class intentionally doesn't modify original code; it loads the model
    weights from `models/smile_cnn_best.pth` by default.
    """

    def __init__(self, model_path: str = None, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        # Use ensemble detector (Haar + DNN + MTCNN) for robust multi-angle detection
        self.detector = EnsembleFaceDetector(min_face_size=15, iou_threshold=0.25)

        # load classifier model
        self.model = None
        if model_path is None:
            model_path = os.path.join(ROOT, 'models', 'smile_cnn_best.pth')
        if SmileCNN is None:
            # cannot import model class; we'll attempt to load scripted model as fallback
            if os.path.exists(model_path):
                try:
                    self.model = torch.jit.load(model_path, map_location=self.device)
                except Exception:
                    self.model = None
        else:
            try:
                self.model = SmileCNN()
                state = torch.load(model_path, map_location=self.device)
                # if state is a dict with key 'model' or similar, try to find
                if isinstance(state, dict) and 'state_dict' in state:
                    self.model.load_state_dict(state['state_dict'])
                elif isinstance(state, dict) and 'model_state_dict' in state:
                    self.model.load_state_dict(state['model_state_dict'])
                elif isinstance(state, dict) and all(k.startswith('module.') for k in state.keys()):
                    # strip module.
                    from collections import OrderedDict
                    new_state = OrderedDict()
                    for k, v in state.items():
                        new_state[k.replace('module.', '')] = v
                    self.model.load_state_dict(new_state)
                else:
                    try:
                        self.model.load_state_dict(state)
                    except Exception:
                        # maybe it's scripted
                        self.model = torch.jit.load(model_path, map_location=self.device)
                if hasattr(self.model, 'to'):
                    self.model.to(self.device)
                if hasattr(self.model, 'eval'):
                    self.model.eval()
            except FileNotFoundError:
                self.model = None
            except Exception:
                # fallback
                try:
                    self.model = torch.jit.load(model_path, map_location=self.device)
                except Exception:
                    self.model = None

        # transform for classifier input (grayscale 1 channel assumed)
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def analyze_pil(self, pil_image: Image.Image, min_confidence: float = 0.3) -> List[Dict]:
        """Detect faces and run smile classifier.

        Returns list of dicts with keys: bbox, face_score, smile_score, label
        """
        import cv2
        import numpy as np
        
        # Convert PIL to OpenCV BGR format
        image_np = np.array(pil_image)
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
        
        results = []
        faces = self.detector.detect(image_bgr)
        
        for x, y, w, h, face_score in faces:
            if face_score < min_confidence:
                # Still process low-confidence faces
                pass
            
            # Crop face region
            x2, y2 = x + w, y + h
            crop_bgr = image_bgr[y:y2, x:x2]
            if crop_bgr.size == 0:
                continue
            
            # Convert crop to PIL for transform
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            crop_pil = Image.fromarray(crop_rgb)
            
            input_tensor = self.transform(crop_pil).unsqueeze(0).to(self.device)
            smile_score = None
            label = 'unknown'
            
            if self.model is not None:
                try:
                    with torch.no_grad():
                        out = self.model(input_tensor)
                        # out may be logits or a single value; try to reduce
                        if isinstance(out, torch.Tensor):
                            if out.numel() == 1:
                                score = float(torch.sigmoid(out).item())
                            elif out.size(1) == 1:
                                score = float(torch.sigmoid(out).item())
                            else:
                                # assume two-class logits
                                probs = torch.softmax(out, dim=1)
                                # index 1 -> smile
                                score = float(probs[0, 1].item())
                            smile_score = score
                            label = 'Cuoi' if smile_score >= 0.5 else 'Binh thuong'
                except Exception:
                    smile_score = None
                    label = 'error'

            results.append({
                'bbox': {'x1': int(x), 'y1': int(y), 'x2': int(x2), 'y2': int(y2)},
                'face_score': float(face_score),
                'smile_score': None if smile_score is None else float(smile_score),
                'label': label
            })

        return results

    def analyze_file(self, filepath: str, **kwargs) -> List[Dict]:
        pil = Image.open(filepath).convert('RGB')
        return self.analyze_pil(pil, **kwargs)

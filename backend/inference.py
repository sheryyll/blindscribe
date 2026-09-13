"""
Loads the exported ONNX model once and exposes a simple predict_char() API.
"""

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

try:
    from .preprocessing import normalize_char_crop
except ImportError:
    from preprocessing import normalize_char_crop

MODEL_PATH = Path(__file__).parent / "model.onnx"
CLASSES_PATH = Path(__file__).parent / "model_classes.json"

TOP_K = 3


class CharacterClassifier:
    def __init__(self, model_path: Path = MODEL_PATH, classes_path: Path = CLASSES_PATH):
        if not model_path.exists():
            raise FileNotFoundError(
                f"{model_path} not found. Run model/train.py then "
                f"model/export_onnx.py first (see README)."
            )
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        with open(classes_path) as f:
            self.classes = json.load(f)

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - np.max(x))
        return e / e.sum()

    def predict_char(self, crop: Image.Image) -> dict:
        """
        Returns:
            {
              "char": "H",
              "confidence": 0.97,
              "top_alternatives": [{"char": "H", "confidence": 0.97}, ...]
            }
        """
        arr = normalize_char_crop(crop)          # (1, 28, 28)
        batch = arr[np.newaxis, ...].astype(np.float32)  # (1, 1, 28, 28)

        logits = self.session.run(None, {self.input_name: batch})[0][0]  # (num_classes,)
        probs = self._softmax(logits)

        top_idx = np.argsort(probs)[::-1][:TOP_K]
        alternatives = [
            {"char": self.classes[i], "confidence": round(float(probs[i]), 4)} for i in top_idx
        ]

        return {
            "char": alternatives[0]["char"],
            "confidence": alternatives[0]["confidence"],
            "top_alternatives": alternatives,
        }

    def predict_string(self, crops: list[Image.Image]) -> list[dict]:
        return [self.predict_char(crop) for crop in crops]


# Loaded once at module import time (FastAPI startup), not per-request.
_classifier: CharacterClassifier | None = None


def get_classifier() -> CharacterClassifier:
    global _classifier
    if _classifier is None:
        _classifier = CharacterClassifier()
    return _classifier

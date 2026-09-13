"""
Grad-CAM explainability for a single character crop.

This is intentionally kept separate from inference.py: the main prediction
path uses ONNX Runtime for a small, fast-starting service, but Grad-CAM
needs gradients, so it loads the original torch checkpoint directly. This
module is only imported/used by the /explain endpoint, so a deployment that
doesn't need explainability can skip installing torch entirely.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.append(str(Path(__file__).parent.parent / "model"))
from train import CharCNN  # noqa: E402

try:
    from .preprocessing import normalize_char_crop
except ImportError:
    from preprocessing import normalize_char_crop

CHECKPOINT_PATH = Path(__file__).parent / "checkpoint.pt"


class GradCAMExplainer:
    def __init__(self, checkpoint_path: Path = CHECKPOINT_PATH):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        self.classes = ckpt["classes"]
        self.model = CharCNN(num_classes=len(self.classes))
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

    def explain(self, crop: Image.Image) -> dict:
        """
        Returns the predicted char plus a (28, 28) heatmap in [0, 1]
        indicating which pixels most influenced the prediction.
        """
        arr = normalize_char_crop(crop)  # (1, 28, 28)
        x = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)  # (1, 1, 28, 28)
        x.requires_grad_(False)

        # Single forward pass that keeps the last conv feature map attached
        # to the autograd graph (see _forward_with_features), since that's
        # the feature map Grad-CAM needs gradients on.
        logits = self._forward_with_features(x)
        pred_idx = int(logits.argmax(dim=1).item())
        features = self._last_features  # (1, C, H, W)

        self.model.zero_grad()
        logits[0, pred_idx].backward()

        grads = features.grad[0]        # (C, H, W)
        acts = features.detach()[0]     # (C, H, W)
        weights = grads.mean(dim=(1, 2))  # (C,) - global-average-pooled gradients

        cam = torch.zeros(acts.shape[1:])
        for c, w in enumerate(weights):
            cam += w * acts[c]
        cam = F.relu(cam)
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        heatmap = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0), size=(28, 28), mode="bilinear", align_corners=False
        )[0, 0]

        return {
            "char": self.classes[pred_idx],
            "heatmap": heatmap.detach().numpy().tolist(),
        }

    def _forward_with_features(self, x):
        """Replicates CharCNN.forward but keeps the intermediate feature
        map (post conv3, pre-pool) in the graph so we can backprop into it."""
        m = self.model
        x = m.pool(F.relu(m.conv1(x)))
        x = m.pool(F.relu(m.conv2(x)))
        feat = F.relu(m.conv3(x))
        feat.retain_grad()
        self._last_features = feat
        pooled = m.pool(feat)
        flat = pooled.flatten(1)
        h = F.relu(m.fc1(flat))
        return m.fc2(h)


_explainer: "GradCAMExplainer | None" = None


def get_explainer() -> GradCAMExplainer:
    global _explainer
    if _explainer is None:
        _explainer = GradCAMExplainer()
    return _explainer

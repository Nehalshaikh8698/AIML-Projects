import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule",
    "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema",
    "Fibrosis", "Pleural Thickening", "Hernia",
]


class ModelService:
    def __init__(self, checkpoint_path: Path, threshold: float = 0.50) -> None:
        self.checkpoint_path = checkpoint_path
        self.threshold = threshold
        self.device = "not-loaded"
        self.torch = None
        self.functional = None
        self.model = None
        self.activation = None
        self.gradient = None
        self.transform = None
        self.load_error: str | None = None
        self.load()

    @property
    def is_ready(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        if not self.checkpoint_path.exists():
            self.load_error = f"Trained model checkpoint not found at {self.checkpoint_path}. Train ml/train.py first."
            logger.warning(self.load_error)
            return
        try:
            import torch
            import torch.nn.functional as functional
            from torchvision import transforms
            from torchvision.models import densenet121
            self.torch = torch
            self.functional = functional
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.transform = transforms.Compose([
                transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            model = densenet121(weights=None)
            model.classifier = torch.nn.Linear(model.classifier.in_features, len(LABELS))
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
            state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
            model.load_state_dict(state_dict)
            model.to(self.device).eval()
            self.model = model
            self.model.features.denseblock4.register_forward_hook(self._save_activation)
            self.load_error = None
            logger.info("Loaded model checkpoint from %s on %s", self.checkpoint_path, self.device)
        except Exception as exc:
            self.load_error = f"Model checkpoint could not be loaded: {exc}"
            logger.exception(self.load_error)

    def _save_activation(self, _module, _inputs, output):
        self.activation = output
        if output.requires_grad:
            output.register_hook(self._save_gradient)

    def _save_gradient(self, gradient):
        self.gradient = gradient

    def predict(self, image: Image.Image) -> dict[str, Any]:
        if not self.is_ready:
            raise RuntimeError(self.load_error or "Model is not ready.")
        x = self.transform(image).unsqueeze(0).to(self.device)
        self.model.zero_grad(set_to_none=True)
        torch = self.torch
        with torch.enable_grad():
            logits = self.model(x)
            probabilities = torch.sigmoid(logits)[0]
            top_index = int(torch.argmax(probabilities).item())
            score = float(probabilities[top_index].detach().cpu().item())
            target_logit = logits[0, top_index]
            target_logit.backward()

        scores = probabilities.detach().cpu().numpy().astype(float)
        predictions = [
            {"label": label, "score": round(float(value), 6), "above_threshold": bool(value >= self.threshold)}
            for label, value in zip(LABELS, scores)
        ]
        predictions.sort(key=lambda item: item["score"], reverse=True)
        cam = self._make_gradcam(image.size)
        return {
            "top_finding": LABELS[top_index],
            "top_score": round(score, 6),
            "threshold": self.threshold,
            "predictions": predictions,
            "gradcam": cam,
            "model_name": "DenseNet121-NIH14",
            "device": str(self.device),
            "notice": "Experimental model output for education only; not a diagnosis.",
        }

    def _make_gradcam(self, original_size: tuple[int, int]) -> Image.Image:
        if self.activation is None or self.gradient is None:
            raise RuntimeError("Grad-CAM activations were not captured.")
        torch = self.torch
        weights = self.gradient.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activation).sum(dim=1, keepdim=True))
        cam = self.functional.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)[0, 0]
        cam = cam.detach().cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        # Simple heat palette avoids an extra plotting dependency.
        r = np.clip(1.5 * cam, 0, 1)
        g = np.clip(1.5 - np.abs(2 * cam - 1) * 1.5, 0, 1)
        b = np.clip(1.0 - 1.5 * cam, 0, 1)
        rgb = np.stack([r, g, b], axis=-1)
        heat = Image.fromarray((rgb * 255).astype(np.uint8), mode="RGB")
        return heat.resize(original_size, Image.Resampling.BILINEAR)

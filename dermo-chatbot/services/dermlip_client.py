"""High-level wrapper for DermLIP visual predictions from base64/path images."""

import base64
import io
from pathlib import Path
from typing import Optional

from PIL import Image

from services.dermlip_model import DermLIPClassifier

# Singleton — loaded once at startup via init_dermlip()
_classifier: Optional[DermLIPClassifier] = None


def init_dermlip() -> None:
    """Load the DermLIP model into memory. Call once at app startup."""
    global _classifier
    if _classifier is None:
        print("Loading DermLIP model...")
        _classifier = DermLIPClassifier()
        print("DermLIP model loaded successfully.")


def is_loaded() -> bool:
    return _classifier is not None


def predict_from_b64(image_b64: str, top_k: int = 5) -> list[dict]:
    """Run DermLIP prediction on a base64-encoded image."""
    if _classifier is None:
        raise RuntimeError("DermLIP model not loaded. Call init_dermlip() first.")
    raw = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    return _classifier.predict(image, top_k=top_k)


def predict_from_path(image_path: str, top_k: int = 5) -> list[dict]:
    """Run DermLIP prediction on an image file path."""
    if _classifier is None:
        raise RuntimeError("DermLIP model not loaded. Call init_dermlip() first.")
    image = Image.open(Path(image_path)).convert("RGB")
    return _classifier.predict(image, top_k=top_k)

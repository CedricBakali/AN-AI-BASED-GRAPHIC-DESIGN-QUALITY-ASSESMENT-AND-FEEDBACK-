"""Image pre-processing (Module 3).

Reads an uploaded design once and prepares every colour space the analysis
and AI feature steps need. The original uploaded file is never modified.
"""
from __future__ import annotations

import numpy as np
import cv2
from PIL import Image

# Keep images modest so analysis stays fast and memory stays small.
MAX_SIDE = 1000


def load_rgb(path: str) -> np.ndarray:
    """Open an image and return an RGB uint8 array, downscaled if huge."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = min(1.0, MAX_SIDE / max(w, h))
    if scale < 1.0:
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))
    return np.asarray(img, dtype=np.uint8)


def to_gray(rgb: np.ndarray) -> np.ndarray:
    """8-bit grayscale version of an RGB array."""
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def to_hsv(rgb: np.ndarray) -> np.ndarray:
    """HSV version of an RGB array (H in 0-180 for OpenCV)."""
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)


def image_properties(path: str) -> dict:
    """Basic facts about the file: size, format, orientation."""
    with Image.open(path) as im:
        w, h = im.size
        return {
            "width": w,
            "height": h,
            "ratio": round(w / h, 3) if h else 0,
            "format": (im.format or "").upper(),
            "is_portrait": h > w,
            "is_landscape": w > h,
            "megapixels": round((w * h) / 1_000_000, 2),
        }


def prepare(path: str) -> dict:
    """One call that returns the loaded image plus its prepared views."""
    rgb = load_rgb(path)
    gray = to_gray(rgb)
    hsv = to_hsv(rgb)
    return {
        "rgb": rgb,
        "gray": gray,
        "hsv": hsv,
        "properties": image_properties(path),
    }

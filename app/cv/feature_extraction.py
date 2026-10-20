"""Feature extraction for the AI aesthetic model.

Turns one design image into a fixed-length list of numbers (the "feature
vector"). Every feature has a plain-English name, a bounded 0-1 range, and a
clear meaning — so a prediction can always be explained back to the student.

These are measurable visual facts (brightness, contrast, edge density,
balance, symmetry...). They are the INPUTS to the trained model, not scores.
"""
from __future__ import annotations

import cv2
import numpy as np

# Stable order — this list IS the model's input order. Never reorder silently.
FEATURE_NAMES = [
    "brightness",        # mean lightness (0 = dark, 1 = bright)
    "contrast_rms",      # how much light/dark variation exists
    "colorfulness",      # Hasler-Susstrunk colourfulness
    "saturation_mean",   # average colour intensity
    "edge_density",      # share of pixels that are edges (detail level)
    "white_space",       # 1 - edge_density (breathing room)
    "thirds_fit",        # share of visual mass near rule-of-thirds lines
    "symmetry",          # left half mirrors right half (0-1)
    "balance",           # even spread of detail across the 4 quadrants
    "text_density",      # share of the canvas taken by text-like regions
    "contrast_ratio",    # WCAG ratio of the 2 dominant colours, /21
    "hue_variety",       # how many different hues are in play (0-1)
]


def _edges(gray8: np.ndarray) -> np.ndarray:
    return cv2.Canny(gray8, 80, 180)


def _dominant_contrast(rgb: np.ndarray) -> float:
    """WCAG contrast between the two dominant colours, normalised to 0-1."""
    pixels = rgb.reshape(-1, 3).astype(np.float32)
    if len(pixels) > 40000:
        idx = np.random.default_rng(42).choice(len(pixels), 40000, replace=False)
        pixels = pixels[idx]
    k = 3
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, _, centers = cv2.kmeans(pixels, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    # The most saturated of the top-2 clusters vs the other: proxy for
    # "loud colour against quiet colour".
    centers = sorted([c for c in centers], key=lambda c: float(np.max(c)))[::-1]
    c1, c2 = centers[0], centers[1]

    def lum(c):
        srgb = c.astype(np.float64) / 255.0
        lin = np.where(srgb <= 0.03928, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
        return float(0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2])

    l1, l2 = lum(c1), lum(c2)
    lighter, darker = max(l1, l2), min(l1, l2)
    ratio = (lighter + 0.05) / (darker + 0.05)
    return float((ratio - 1.0) / 20.0)  # 1:1 -> 0, 21:1 -> 1


def _text_density(gray8: np.ndarray) -> float:
    """Share of the canvas covered by small text-like regions (MSER)."""
    try:
        mser = cv2.MSER_create()
        regions, _ = mser.detectRegions(gray8)
    except Exception:
        return 0.0
    if not regions:
        return 0.0
    h, w = gray8.shape
    total = h * w
    area = 0.0
    for r in regions[:3000]:
        x, y, bw, bh = cv2.boundingRect(r.reshape(-1, 1, 2))
        if 5 < bh < h * 0.5 and 0.1 < bw / (bh + 1e-6) < 15:
            area += bw * bh
    return float(min(1.0, area / total))


def extract_features(img: np.ndarray) -> dict:
    """Compute all 12 features for an RGB image. Returns {name: value}."""
    gray8 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = gray8.astype(np.float64) / 255.0
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    brightness = float(gray.mean())
    contrast_rms = float(gray.std())

    # Hasler-Susstrunk colourfulness, scaled to a friendly 0-1 range.
    r, g, b = [img[:, :, i].astype(np.float64) for i in range(3)]
    rg = r - g
    yb = 0.5 * (r + g) - b
    cf = np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    colorfulness = float(min(1.0, cf / 150.0))

    saturation = float((hsv[:, :, 1].astype(np.float64) / 255.0).mean())

    edges = _edges(gray8)
    edge_density = float((edges > 0).mean())
    white_space = 1.0 - edge_density

    # --- balance across quadrants ------------------------------------
    h, w = edges.shape
    quad = [
        edges[: h // 2, : w // 2].mean(),
        edges[: h // 2, w // 2:].mean(),
        edges[h // 2:, : w // 2].mean(),
        edges[h // 2:, w // 2:].mean(),
    ]
    q = np.array(quad) + 1e-6
    balance = float(max(0.0, min(1.0, 1.0 - (q.std() / q.mean()) / 2.0)))

    # --- thirds fit: how much detail sits near the 4 thirds intersections
    mass = cv2.dilate(edges, np.ones((7, 7), np.uint8)).astype(np.float64)
    if mass.sum() > 0:
        ys, xs = np.nonzero(mass)
        pts = np.stack([xs / w, ys / h], axis=1)
        thirds = np.array([[1 / 3, 1 / 3], [2 / 3, 1 / 3],
                           [1 / 3, 2 / 3], [2 / 3, 2 / 3]])
        dist = np.linalg.norm(pts[:, None, :] - thirds[None, :, :], axis=2).min(axis=1)
        thirds_fit = float((dist < 0.20).mean())
    else:
        thirds_fit = 0.0

    # --- symmetry: left half vs mirrored right half --------------------
    half = w // 2
    if half > 4:
        left = gray[:, :half]
        right = np.fliplr(gray)[:, :half]
        symmetry = float(max(0.0, 1.0 - np.abs(left - right).mean() * 4.0))
    else:
        symmetry = 0.0

    # --- hue variety: spread of saturated pixels around the colour wheel
    sat_mask = hsv[:, :, 1] > 40
    if sat_mask.sum() > 100:
        hues = hsv[:, :, 0][sat_mask].astype(np.float64) * 2.0  # back to degrees
        # Circular spread -> 0 (one hue) .. 1 (everywhere).
        hue_variety = float(np.sqrt(1.0 - abs(np.exp(1j * np.deg2rad(hues)).mean())))
        hue_variety = float(min(1.0, hue_variety / 0.9))
    else:
        hue_variety = 0.0

    text_density = _text_density(gray8)
    contrast_ratio = _dominant_contrast(img)

    values = {
        "brightness": round(brightness, 4),
        "contrast_rms": round(contrast_rms, 4),
        "colorfulness": round(colorfulness, 4),
        "saturation_mean": round(saturation, 4),
        "edge_density": round(edge_density, 4),
        "white_space": round(white_space, 4),
        "thirds_fit": round(thirds_fit, 4),
        "symmetry": round(symmetry, 4),
        "balance": round(balance, 4),
        "text_density": round(text_density, 4),
        "contrast_ratio": round(contrast_ratio, 4),
        "hue_variety": round(hue_variety, 4),
    }
    # Guarantee the exact order and bounds the model expects.
    return {name: float(min(1.0, max(0.0, values[name]))) for name in FEATURE_NAMES}


def to_vector(features: dict) -> np.ndarray:
    """Turn {name: value} into the ordered 1-D array the model consumes."""
    return np.array([features[name] for name in FEATURE_NAMES], dtype=np.float64)

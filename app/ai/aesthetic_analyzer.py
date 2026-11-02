"""EXPERIMENTAL / LEGACY - scikit-learn labelling experiment.

.. warning::
   **This module is no longer called by the ``/analyze`` runtime
   pipeline.** It is kept so the earlier experiment remains reproducible.
   The AI-Assisted Assessment on reports now comes from the pre-trained
   CLIP model in :mod:`app.ai.clip_analyzer`.

Legacy pipeline position:
    IMAGE -> PRE-PROCESSING -> CV FEATURES -> TRAINED MODEL -> AI SCORE
"""
from __future__ import annotations

from app.ai.model_manager import (
    ModelInfo,
    load_model,
    predict_score,
    top_factors,
)
from app.cv.feature_extraction import extract_features
from app.cv.preprocessing import prepare

# Human-readable names for the 12 features, used in the report.
FEATURE_LABELS = {
    "brightness": "Brightness",
    "contrast_rms": "Contrast",
    "colorfulness": "Colourfulness",
    "saturation_mean": "Saturation",
    "edge_density": "Detail level",
    "white_space": "White space",
    "thirds_fit": "Rule-of-thirds fit",
    "symmetry": "Symmetry",
    "balance": "Balance",
    "text_density": "Text coverage",
    "contrast_ratio": "Colour contrast ratio",
    "hue_variety": "Hue variety",
}


def _describe(feature: str, value: float) -> str:
    """Plain-English reading of one feature value."""
    if feature == "brightness":
        return "very dark" if value < 0.3 else ("bright" if value > 0.7 else "mid-tone")
    if feature == "white_space":
        return "lots of empty space" if value > 0.7 else (
            "very busy, little empty space" if value < 0.3 else "moderate spacing")
    if feature == "symmetry":
        return "strongly symmetrical" if value > 0.6 else (
            "asymmetrical" if value < 0.3 else "partly symmetrical")
    if feature == "colorfulness":
        return "muted, near-monochrome" if value < 0.15 else (
            "very colourful" if value > 0.6 else "moderately colourful")
    if feature == "contrast_rms":
        return "strong light/dark contrast" if value > 0.35 else (
            "flat, low contrast" if value < 0.15 else "moderate contrast")
    if feature == "hue_variety":
        return "few hues in play" if value < 0.25 else (
            "many hues in play" if value > 0.6 else "a balanced hue range")
    return f"{value:.2f}"


def analyze_aesthetics(image_path: str) -> dict | None:
    """
    Predict the aesthetic quality of one design.

    Returns:
        {
            "score":        int  0-100 model estimate,
            "mae":          float typical error in points (honesty figure),
            "samples":      int   labelled designs the model learned from,
            "factors":      list  the features that mattered most here,
            "trained_at":   str   when the model was trained,
        }

        None when no model has been trained yet — callers should then fall
        back to the rule-based report only.
    """
    model, info = load_model()
    if model is None:
        return None

    data = prepare(image_path)
    features = extract_features(data["rgb"])

    score = predict_score(model, features)

    factors = []
    for item in top_factors(model, features, limit=3):
        factors.append({
            "name": FEATURE_LABELS.get(item["feature"], item["feature"]),
            "value": item["value"],
            "reading": _describe(item["feature"], item["value"]),
            "weight": item["weight"],
        })

    return {
        "score": int(round(score)),
        "mae": info.mae if info else None,
        "samples": info.n_samples if info else 0,
        "factors": factors,
        "trained_at": info.trained_at if info else None,
    }


def model_info() -> ModelInfo | None:
    """Metadata about the current model, or None if untrained."""
    _, info = load_model()
    return info

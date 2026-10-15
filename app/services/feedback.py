"""Feedback service (Module 6).

Converts deterministic category scores from app/services/analysis.py into
student-friendly feedback:

- strengths:      categories scoring >= STRENGTH_THRESHOLD
- weaknesses:     categories scoring <  WEAKNESS_THRESHOLD
- recommendations: actionable suggestions, weakest category first

No ML here. Thresholds are fixed and explainable so every sentence on the
result page can be traced back to a score.
"""
from __future__ import annotations

STRENGTH_THRESHOLD = 75
WEAKNESS_THRESHOLD = 60


def build_feedback(categories: list[dict]) -> dict:
    """Split category results into strengths / weaknesses / recommendations.

    Args:
        categories: list of {"name": str, "score": int,
                    "summary": str, "suggestions": [str]}
    Returns:
        {"strengths": [...], "weaknesses": [...], "recommendations": [...]}
    """
    strengths: list[dict] = []
    weaknesses: list[dict] = []

    for cat in categories:
        entry = {
            "name": cat.get("name", "?"),
            "score": int(cat.get("score", 0)),
            "summary": cat.get("summary", ""),
        }
        if entry["score"] >= STRENGTH_THRESHOLD:
            strengths.append(entry)
        elif entry["score"] < WEAKNESS_THRESHOLD:
            weaknesses.append(entry)

    # Recommendations come from the stored per-category suggestions,
    # ordered weakest-first so the biggest problem is fixed first.
    recommendations: list[dict] = []
    for cat in sorted(categories, key=lambda c: int(c.get("score", 0))):
        for text in cat.get("suggestions", []) or []:
            recommendations.append(
                {"category": cat.get("name", "?"), "text": text}
            )

    return {
        "strengths": strengths,
        "weaknesses": weaknesses,
        "recommendations": recommendations,
    }

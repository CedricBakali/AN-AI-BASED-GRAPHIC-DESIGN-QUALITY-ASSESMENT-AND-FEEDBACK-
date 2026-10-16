"""Tests for explainable scoring + feedback split (Modules 5-6)."""
from __future__ import annotations

from app.services.feedback import (
    WEAKNESS_THRESHOLD,
    STRENGTH_THRESHOLD,
    build_feedback,
)
from app.services.scoring import compute_overall, grade_for


def test_compute_overall_is_mean():
    assert compute_overall([80, 70, 90]) == 80
    assert compute_overall([]) == 0


def test_grade_bands():
    assert grade_for(95) == "A - Excellent"
    assert grade_for(90) == "A - Excellent"
    assert grade_for(89) == "B - Good"
    assert grade_for(80) == "B - Good"
    assert grade_for(79) == "C - Fair"
    assert grade_for(70) == "C - Fair"
    assert grade_for(69) == "D - Needs work"
    assert grade_for(60) == "D - Needs work"
    assert grade_for(59) == "E - Poor"
    assert grade_for(0) == "E - Poor"


def _cat(name, score):
    return {"name": name, "score": score,
            "summary": f"{name} summary", "suggestions": [f"fix {name}"]}


def test_feedback_split():
    fb = build_feedback([_cat("Contrast", 90), _cat("Alignment", 40),
                         _cat("Layout", 70)])
    assert [s["name"] for s in fb["strengths"]] == ["Contrast"]
    assert [w["name"] for w in fb["weaknesses"]] == ["Alignment"]
    # neutral middle band (60-74) is neither strength nor weakness
    assert all(s["name"] != "Layout" for s in fb["strengths"])
    assert all(w["name"] != "Layout" for w in fb["weaknesses"])


def test_recommendations_weakest_first():
    fb = build_feedback([_cat("Contrast", 90), _cat("Alignment", 40)])
    assert fb["recommendations"][0] == {"category": "Alignment",
                                       "text": "fix Alignment"}
    assert fb["recommendations"][-1]["category"] == "Contrast"


def test_threshold_constants_sane():
    assert 0 < WEAKNESS_THRESHOLD < STRENGTH_THRESHOLD <= 100

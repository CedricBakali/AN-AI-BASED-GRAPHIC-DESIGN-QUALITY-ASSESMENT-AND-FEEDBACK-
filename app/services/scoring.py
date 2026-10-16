"""Quality scoring (Module 5).

Explainable aggregation of category scores. No ML: the overall score is
the plain arithmetic mean of the category scores, and the grade is a
fixed band mapping. Anyone can recompute both by hand from the result page.
"""
from __future__ import annotations


def compute_overall(scores: list[int]) -> int:
    """Mean of category scores (0-100 each), rounded to an int."""
    if not scores:
        return 0
    return int(round(sum(scores) / len(scores)))


def grade_for(score: int) -> str:
    """Fixed grade bands for an overall score."""
    if score >= 90:
        return "A - Excellent"
    if score >= 80:
        return "B - Good"
    if score >= 70:
        return "C - Fair"
    if score >= 60:
        return "D - Needs work"
    return "E - Poor"

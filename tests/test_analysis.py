"""Tests for the deterministic analysis pipeline (Module 4)."""
from __future__ import annotations

import io

from PIL import Image, ImageDraw

from app.services.analysis import (
    analyze_image,
    score_layout,
    score_visual_hierarchy,
    load_image,
)


def _poster(path, size=(480, 640)):
    """Build a posteresque image: dark bg, light headline bar, accent box."""
    img = Image.new("RGB", size, (18, 24, 40))
    d = ImageDraw.Draw(img)
    d.rectangle([40, 50, 440, 150], fill=(245, 240, 225))   # headline band
    d.rectangle([40, 200, 280, 420], fill=(200, 60, 50))    # hero block
    d.rectangle([300, 200, 440, 300], fill=(240, 180, 40))  # accent
    for y in range(460, 580, 24):                            # body lines
        d.rectangle([40, y, 440, y + 10], fill=(120, 130, 150))
    img.save(path)
    return str(path)


def test_analyze_image_returns_eight_categories(tmp_path):
    result = analyze_image(_poster(tmp_path / "p.png"))
    names = [c.name for c in result.categories]
    assert names == ["Contrast", "Color Harmony", "Typography", "Alignment",
                     "White Space", "Readability", "Layout", "Visual Hierarchy"]
    for c in result.categories:
        assert 0 <= c.score <= 100
        assert c.summary
        assert c.suggestions  # every score must be explainable
    assert 0 <= result.overall <= 100
    assert result.grade in ("A - Excellent", "B - Good", "C - Fair",
                            "D - Needs work", "E - Poor")
    assert result.palette  # at least one dominant colour


def test_overall_is_mean_of_categories(tmp_path):
    result = analyze_image(_poster(tmp_path / "p.png"))
    expected = round(sum(c.score for c in result.categories) / 8)
    assert result.overall == expected


def test_blank_image_stays_neutral(tmp_path):
    Image.new("RGB", (400, 400), (200, 200, 200)).save(tmp_path / "b.png")
    img = load_image(str(tmp_path / "b.png"))
    assert 0 <= score_layout(img).score <= 100
    assert 0 <= score_visual_hierarchy(img).score <= 100


def test_small_image_bytes_analyze(tmp_path):
    buf = io.BytesIO()
    Image.new("RGB", (300, 200), (30, 120, 200)).save(buf, format="PNG")
    p = tmp_path / "s.png"
    p.write_bytes(buf.getvalue())
    assert analyze_image(str(p)).overall >= 0

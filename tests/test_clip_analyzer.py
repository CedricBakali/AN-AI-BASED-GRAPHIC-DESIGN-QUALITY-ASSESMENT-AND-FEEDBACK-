"""Tests for the runtime AI: zero-shot CLIP inference (app.ai.clip_analyzer).

The model is 605 MB and costs ~30 s to import on this machine, so the real
inference test is guarded by a single session-scoped fixture that loads it
once and reuses it for every test in this file.

Tests that must be fast and deterministic stub ``model_loader.load_clip``.
"""
from __future__ import annotations

import os

import pytest

from app.ai import clip_analyzer, model_loader

REAL_UPLOADS = os.path.join("static", "uploads")

# The exact contract every caller (and the documentation) depends on.
REQUIRED_KEYS = {"score", "model", "available", "explanation"}


def _a_real_design() -> str | None:
    """A reasonably large real uploaded design, or None."""
    if not os.path.isdir(REAL_UPLOADS):
        return None
    candidates = [
        os.path.join(REAL_UPLOADS, name)
        for name in sorted(os.listdir(REAL_UPLOADS))
        if name.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
        and os.path.getsize(os.path.join(REAL_UPLOADS, name)) > 40_000
    ]
    if not candidates:
        return None
    return max(candidates, key=os.path.getsize)


# ---------------------------------------------------------------
# unavailable path (fast - no model load, no network)
# ---------------------------------------------------------------

def test_analyzer_reports_unavailable_when_model_cannot_load(
        monkeypatch, tmp_path):
    """Missing weights + offline -> honest 'unavailable', no exception."""
    monkeypatch.setattr(model_loader, "load_clip", lambda: None)
    monkeypatch.setattr(
        model_loader, "status",
        lambda: {"reason": "Pre-trained CLIP weights are unavailable."})

    result = clip_analyzer.analyze_with_ai("static/uploads/anything.png")

    assert REQUIRED_KEYS <= set(result)
    assert result["available"] is False
    assert result["score"] is None
    assert "unavailable" in result["explanation"].lower()
    assert result["model"] == "openai/clip-vit-base-patch32"


def test_unavailable_result_never_raises_on_a_missing_file(monkeypatch):
    """Even a path that does not exist must return a dict, not raise."""
    monkeypatch.setattr(model_loader, "load_clip", lambda: None)
    monkeypatch.setattr(model_loader, "status", lambda: {"reason": "no model"})

    result = clip_analyzer.analyze_with_ai(
        "static/uploads/definitely-not-here.png")
    assert result["available"] is False
    assert result["score"] is None


def test_module_import_is_lazy():
    """Importing clip_analyzer must not pull torch/transformers in."""
    import subprocess
    import sys

    code = (
        "import sys; import app.ai.clip_analyzer, app.ai.model_loader; "
        "print('LEAKED' if any(m in sys.modules for m in "
        "('torch','transformers')) else 'CLEAN')"
    )
    out = subprocess.run([sys.executable, "-c", code],
                         capture_output=True, text=True, timeout=180)
    assert "CLEAN" in out.stdout, out.stdout + out.stderr


# ---------------------------------------------------------------
# available path (real inference - loaded once for this file)
# ---------------------------------------------------------------

@pytest.fixture(scope="module")
def real_design():
    path = _a_real_design()
    if path is None:
        pytest.skip("no uploaded design large enough to test against")
    return path


@pytest.fixture(scope="module")
def loaded_model():
    """Load the real pre-trained CLIP model once for this module."""
    status = model_loader.warmup()
    if not status["available"]:
        pytest.skip("CLIP model unavailable: %s" % status["reason"])
    return status


def test_model_loader_marks_itself_available(loaded_model):
    assert loaded_model["available"] is True
    assert loaded_model["model"] == "openai/clip-vit-base-patch32"
    assert loaded_model["device"] == "cpu"
    assert loaded_model["threads"] == 2
    assert loaded_model["weights_ready"] is True


def test_model_loader_loads_once_and_caches(loaded_model):
    """Second call must come from cache, not reload the weights."""
    import time

    started = time.perf_counter()
    bundle = model_loader.load_clip()
    elapsed = time.perf_counter() - started

    assert bundle is not None
    assert elapsed < 0.5, f"cached load took {elapsed:.3f}s - cache is broken"


def test_analyzer_returns_a_real_score(loaded_model, real_design):
    """The headline contract: a genuine 0-100 score from CLIP inference."""
    result = clip_analyzer.analyze_with_ai(real_design)

    assert REQUIRED_KEYS <= set(result)
    assert result["available"] is True
    assert result["model"] == "openai/clip-vit-base-patch32"
    assert isinstance(result["score"], int)
    assert 0 <= result["score"] <= 100
    assert result["inference_seconds"] is not None
    assert result["inference_seconds"] > 0
    assert "zero-shot" in result["explanation"]
    assert "pre-trained" in result["explanation"]
    assert "does not measure typography" in result["explanation"]


def test_score_is_deterministic_for_the_same_image(loaded_model, real_design):
    """Same input, same output - the score is never random."""
    first = clip_analyzer.analyze_with_ai(real_design)
    second = clip_analyzer.analyze_with_ai(real_design)
    assert first["score"] == second["score"]


def test_impressions_are_reported_but_never_scored(loaded_model, real_design):
    """Impressions are extra readings; only one antonym pair sets the score."""
    result = clip_analyzer.analyze_with_ai(real_design)

    assert len(result["impressions"]) == len(clip_analyzer.IMPRESSION_PAIRS)
    for impression in result["impressions"]:
        assert impression["label"]
        assert 0.0 <= impression["positive_probability"] <= 1.0
        assert impression["leaning"] in {"positive", "negative"}
    # The score comes from the single SCORE pair, not from an impression.
    assert abs(result["positive_probability"] * 100 - result["score"]) <= 0.51


def test_ai_status_reports_honestly(loaded_model):
    status = clip_analyzer.ai_status()
    assert status["available"] is True
    assert status["model"] == "openai/clip-vit-base-patch32"
    assert status["device"] == "cpu"
    assert "inference only" in status["method"]


def test_no_cuda_is_used(loaded_model):
    """CPU-only requirement - this machine has no usable GPU."""
    import torch

    assert torch.cuda.is_available() is False

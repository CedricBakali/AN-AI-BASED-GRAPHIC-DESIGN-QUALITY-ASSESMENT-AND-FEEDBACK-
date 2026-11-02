"""Tests for the EXPERIMENTAL scikit-learn labelling experiment.

The model file and database are temporary, so these never overwrite a model
you trained for real.

.. note::
   The scikit-learn model is legacy/experimental - it is NOT the runtime AI.
   The runtime AI is the pre-trained CLIP model, tested in
   tests/test_clip_analyzer.py. The last section of this file tests the
   report WIRING with the CLIP call stubbed out (fast, deterministic).
"""
from __future__ import annotations

import os
import random

import pytest

from app.ai import model_manager
from app.ai.aesthetic_analyzer import analyze_aesthetics
from app.ai.model_manager import (
    MIN_PER_CLASS,
    MIN_TOTAL_SAMPLES,
    ModelInfo,
    TrainingDataError,
    load_model,
    predict_score,
    top_factors,
    train_model,
)
from app.cv.feature_extraction import FEATURE_NAMES

REAL_UPLOADS = os.path.join("static", "uploads")


# ---------------------------------------------------------------
# helpers
# ---------------------------------------------------------------

def _fake_rows(n, seed=7):
    """Deterministic, plausible feature rows (values stay in 0-1)."""
    rng = random.Random(seed)
    rows, targets = [], []
    for _ in range(n):
        row = {name: round(rng.random(), 4) for name in FEATURE_NAMES}
        # Make the target depend on a couple of features so the model has
        # something real to learn, otherwise the test proves nothing.
        target = 2.0 + 6.0 * row["contrast_rms"] + 2.0 * row["balance"]
        rows.append(row)
        targets.append(min(9.5, max(1.0, target)))
    # Guarantee both classes are represented.
    targets[0] = 8.5
    targets[1] = 2.5
    return rows, targets


@pytest.fixture()
def isolated_model(tmp_path, monkeypatch):
    """Point the model files at a temporary directory."""
    monkeypatch.setattr(model_manager, "MODEL_PATH",
                        str(tmp_path / "model.pkl"))
    monkeypatch.setattr(model_manager, "METRICS_PATH",
                        str(tmp_path / "metrics.json"))
    return tmp_path


# ---------------------------------------------------------------
# guards
# ---------------------------------------------------------------

def test_refuses_to_train_on_too_little_data():
    rows, targets = _fake_rows(MIN_TOTAL_SAMPLES - 1)
    with pytest.raises(TrainingDataError):
        train_model(rows, targets)


def test_refuses_to_train_when_only_one_class():
    rows, targets = _fake_rows(12)
    targets = [8.0] * len(targets)          # everything looks "good"
    with pytest.raises(TrainingDataError):
        train_model(rows, targets)


def test_refuses_when_class_counts_are_too_thin():
    rows, targets = _fake_rows(12)
    targets = [8.0] * (len(targets) - 1) + [3.0]   # only one weak example
    with pytest.raises(TrainingDataError):
        train_model(rows, targets)


def test_mismatched_rows_and_targets_rejected():
    rows, targets = _fake_rows(10)
    with pytest.raises(TrainingDataError):
        train_model(rows, targets[:-1], 0)


# ---------------------------------------------------------------
# train / save / load / predict
# ---------------------------------------------------------------

def test_train_save_load_predict_roundtrip(isolated_model):
    rows, targets = _fake_rows(24)
    model, info = train_model(rows, targets)

    assert info.n_samples == 24
    assert info.n_good + info.n_weak == 24
    assert 0.0 <= info.discrimination <= 1.0
    assert set(info.feature_importance) == set(FEATURE_NAMES)

    model_manager.save_model(model, info)

    assert os.path.isfile(model_manager.MODEL_PATH)
    assert os.path.isfile(model_manager.METRICS_PATH)
    assert model_manager.is_available()

    loaded, loaded_info = load_model()
    assert loaded is not None
    assert loaded_info.n_samples == 24
    assert loaded_info.trained_at == info.trained_at

    sample = rows[0]
    score = predict_score(loaded, sample)
    assert 0.0 <= score <= 100.0


def test_prediction_stays_inside_the_scale(isolated_model):
    rows, targets = _fake_rows(20)
    model, _ = train_model(rows, targets)

    extreme_dark = {name: 0.0 for name in FEATURE_NAMES}
    extreme_bright = {name: 1.0 for name in FEATURE_NAMES}

    assert 0.0 <= predict_score(model, extreme_dark) <= 100.0
    assert 0.0 <= predict_score(model, extreme_bright) <= 100.0


def test_top_factors_returns_most_influential(isolated_model):
    rows, targets = _fake_rows(20)
    model, _ = train_model(rows, targets)

    factors = top_factors(model, rows[0], limit=3)
    assert len(factors) == 3
    assert {f["feature"] for f in factors} <= set(FEATURE_NAMES)
    weights = [f["weight"] for f in factors]
    assert weights == sorted(weights, reverse=True)


def test_clear_model_removes_artefacts(isolated_model):
    rows, targets = _fake_rows(20)
    model, info = train_model(rows, targets)
    model_manager.save_model(model, info)
    assert model_manager.is_available()

    assert model_manager.clear_model()
    assert not model_manager.is_available()
    assert load_model() == (None, None)


def test_loading_absent_model_is_not_an_error(isolated_model):
    assert load_model() == (None, None)
    assert not model_manager.is_available()


# ---------------------------------------------------------------
# runtime analyzer
# ---------------------------------------------------------------

def test_analyzer_returns_none_when_untrained(isolated_model):
    assert analyze_aesthetics("does/not/matter.png") is None


def test_analyzer_scores_a_real_design(isolated_model):
    """Train on real features, then score a real uploaded design."""
    files = sorted(
        os.path.join(REAL_UPLOADS, f)
        for f in os.listdir(REAL_UPLOADS)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
        and os.path.getsize(os.path.join(REAL_UPLOADS, f)) > 5 * 1024
    )[:8]
    if len(files) < MIN_PER_CLASS * 2:
        pytest.skip("not enough real uploads to build a training set")

    from app.cv.feature_extraction import extract_features
    from app.cv.preprocessing import prepare

    rows, targets = [], []
    for i, path in enumerate(files):
        rows.append(extract_features(prepare(path)["rgb"]))
        targets.append(8.0 if i % 2 == 0 else 3.0)

    model, info = train_model(rows, targets)
    model_manager.save_model(model, info)

    result = analyze_aesthetics(files[0])
    assert result is not None
    assert 0 <= result["score"] <= 100
    assert result["samples"] == info.n_samples
    assert result["factors"]
    assert {"name", "value", "reading", "weight"} <= set(result["factors"][0])


# ---------------------------------------------------------------
# EXPERIMENTAL scikit-learn model info
# ---------------------------------------------------------------

def test_model_info_reports_honestly(isolated_model):
    rows, targets = _fake_rows(20)
    _, info = train_model(rows, targets)

    assert isinstance(info, ModelInfo)
    assert info.n_samples == 20
    assert info.n_augmented == 0
    assert info.mae >= 0


# ---------------------------------------------------------------
# end to end: upload -> stored -> shown on the report
#
# The runtime AI is the pre-trained CLIP model. Loading the real 605 MB
# model inside every test would cost ~30 s and ~1 GB, so these wiring tests
# stub analyze_with_ai and assert what the app DOES with its answer.
# Genuine inference is covered separately in tests/test_clip_analyzer.py.
# ---------------------------------------------------------------

def _png_upload(logged_client, name="design.png", colour=(200, 60, 60)):
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (300, 220), colour).save(buf, format="PNG")
    buf.seek(0)
    return logged_client.post("/analyze", data={"image": (buf, name)},
                              content_type="multipart/form-data",
                              follow_redirects=False)


def _fake_ai(score=73, available=True, factors=None):
    """A stand-in for clip_analyzer.analyze_with_ai's return value."""
    return {
        "score": score,
        "model": "openai/clip-vit-base-patch32",
        "label": "CLIP (OpenAI CLIP)",
        "available": available,
        "explanation": "stub explanation",
        "impressions": (factors if factors is not None else [
            {"label": "Attractiveness", "positive_probability": 0.62},
            {"label": "Professionalism", "positive_probability": 0.41},
            {"label": "Attention-grabbing", "positive_probability": 0.55},
        ]),
        "confidence": 0.73,
        "positive_probability": score / 100.0,
    }


def test_report_shows_ai_panel_when_clip_available(client, logged_client,
                                                   monkeypatch):
    from app.ai import clip_analyzer
    monkeypatch.setattr(clip_analyzer, "analyze_with_ai",
                        lambda _path: _fake_ai(score=73))

    rv = _png_upload(logged_client)
    assert rv.status_code == 302

    html = logged_client.get(rv.location).data.decode()
    assert "AI-Assisted Assessment" in html
    assert "ai-panel" in html
    assert "73/100" in html
    # Model identity is shown honestly.
    assert "openai/clip-vit-base-patch32" in html
    assert "CLIP (OpenAI CLIP)" in html
    assert "Available" in html
    # Scope: the AI is general, the principles are CV.
    assert "general" in html
    assert "separately through computer-vision analysis" in html
    # Regression: HTML entities must never leak as literal text. The
    # explanation is built in Python and then escaped by Jinja, so it has
    # to use real Unicode characters, not &mdash; / &rsquo;.
    assert "&mdash;" not in html
    assert "&rsquo;" not in html
    assert "&amp;mdash;" not in html
    # Regression: the comparison sentence needs its "than".
    assert "points higher the " not in html
    assert "points lower the " not in html


def test_ai_score_and_model_are_stored(client, logged_client, monkeypatch):
    from app.ai import clip_analyzer
    monkeypatch.setattr(clip_analyzer, "analyze_with_ai",
                        lambda _path: _fake_ai(score=61))

    rv = _png_upload(logged_client, "stored.png")
    analysis_id = int(rv.location.rsplit("/", 1)[1])

    from app.models import database
    record = database.get_analysis(analysis_id, session_user_id())
    assert record["ai_score"] == 61
    assert record["ai_model"] == "openai/clip-vit-base-patch32"
    assert isinstance(record["ai_factors"], list)
    assert record["ai_factors"][0]["name"] == "Overall design quality"


def test_report_shows_unavailable_when_clip_is_not(client, logged_client,
                                                   monkeypatch):
    """CLIP unavailable -> ai_score stays NULL and the report says so."""
    from app.ai import clip_analyzer
    monkeypatch.setattr(clip_analyzer, "analyze_with_ai",
                        lambda _path: _fake_ai(score=None, available=False))

    rv = _png_upload(logged_client, "unavailable.png", (60, 120, 200))
    assert rv.status_code == 302

    analysis_id = int(rv.location.rsplit("/", 1)[1])
    from app.models import database
    record = database.get_analysis(analysis_id, session_user_id())
    assert record["ai_score"] is None
    assert record["ai_model"] is None

    html = logged_client.get(rv.location).data.decode()
    assert "AI-Assisted Assessment" in html
    assert "Unavailable" in html
    # The rule-based report must still be complete.
    assert "Category scores" in html


def test_upload_survives_a_broken_clip(client, logged_client, monkeypatch):
    """An AI that raises must never stop the rule-based report."""
    from app.ai import clip_analyzer

    def boom(_path):
        raise RuntimeError("model exploded")

    monkeypatch.setattr(clip_analyzer, "analyze_with_ai", boom)

    rv = _png_upload(logged_client, "safe.png", (90, 90, 90))
    assert rv.status_code == 302
    assert rv.location.startswith("/result/")

    html = logged_client.get(rv.location).data.decode()
    assert "Category scores" in html
    assert "Unavailable" in html


def test_rule_based_score_is_unchanged_by_the_ai(client, logged_client,
                                                 monkeypatch):
    """overall must equal the mean of the category scores, always."""
    from app.ai import clip_analyzer

    for score in (5, 50, 95):
        monkeypatch.setattr(clip_analyzer, "analyze_with_ai",
                            lambda _path, s=score: _fake_ai(score=s))
        rv = _png_upload(logged_client, "unchanged.png", (30, 30 + score, 60))
        assert rv.status_code == 302

        from app.models import database
        record = database.get_analysis(int(rv.location.rsplit("/", 1)[1]),
                                       session_user_id())
        cats = record["categories"]
        mean = round(sum(c["score"] for c in cats) / len(cats))
        assert record["overall"] == mean, "AI must not touch overall"
        assert record["ai_score"] == score


def session_user_id() -> int:
    """The id the test suite's registered user receives in a fresh temp DB."""
    from app.models import database
    with database.get_connection() as conn:
        row = conn.execute(
            "SELECT id FROM users WHERE email = ?",
            ("tester@test.com",)
        ).fetchone()
    assert row is not None, "logged_client must have registered first"
    return int(row["id"])

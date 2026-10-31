"""Tests for the trained aesthetic model (AI Phase 11, step 3).

These run against a temporary model file, so they never overwrite a model
you trained for real, and against a temporary database.
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
# surfaced on the assessment report
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
# ---------------------------------------------------------------

def _train_and_save(isolated_model):
    rows, targets = _fake_rows(20)
    model, info = train_model(rows, targets)
    model_manager.save_model(model, info)
    return info


def test_report_shows_ai_panel_when_model_exists(client, logged_client,
                                                 isolated_model):
    import io

    from PIL import Image

    _train_and_save(isolated_model)

    buf = io.BytesIO()
    Image.new("RGB", (300, 220), (200, 60, 60)).save(buf, format="PNG")
    buf.seek(0)

    rv = logged_client.post("/analyze", data={"image": (buf, "design.png")},
                            content_type="multipart/form-data",
                            follow_redirects=False)
    assert rv.status_code == 302
    assert rv.location.startswith("/result/")

    html = logged_client.get(rv.location).data.decode()
    assert "AI estimate" in html
    assert "ai-panel" in html
    assert "second opinion" in html


def test_report_explains_absent_model(client, logged_client, isolated_model):
    """No model trained yet: the report still renders, and says so."""
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (300, 220), (60, 120, 200)).save(buf, format="PNG")
    buf.seek(0)

    rv = logged_client.post("/analyze", data={"image": (buf, "plain.png")},
                            content_type="multipart/form-data",
                            follow_redirects=False)
    assert rv.status_code == 302

    html = logged_client.get(rv.location).data.decode()
    assert "AI estimate" in html
    assert "no model had been" in html


def test_upload_survives_a_broken_model(client, logged_client, isolated_model,
                                        monkeypatch):
    """A model that raises must never stop the rule-based report."""
    import io

    from PIL import Image

    _train_and_save(isolated_model)

    def boom(_path):
        raise RuntimeError("model exploded")

    monkeypatch.setattr("app.ai.aesthetic_analyzer.analyze_aesthetics", boom)

    buf = io.BytesIO()
    Image.new("RGB", (300, 220), (90, 90, 90)).save(buf, format="PNG")
    buf.seek(0)

    rv = logged_client.post("/analyze", data={"image": (buf, "safe.png")},
                            content_type="multipart/form-data",
                            follow_redirects=False)
    assert rv.status_code == 302
    assert rv.location.startswith("/result/")

    html = logged_client.get(rv.location).data.decode()
    assert "Category scores" in html

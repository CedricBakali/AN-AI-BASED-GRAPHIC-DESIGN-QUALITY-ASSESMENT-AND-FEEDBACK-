"""EXPERIMENTAL / LEGACY - scikit-learn labelling experiment.

.. warning::
   This module is **not** part of the ``/analyze`` runtime pipeline and is
   not the project's AI methodology. It is kept for an offline experiment
   only: an administrator judges designs on the *Label Designs
   (Experimental)* page, and ``scripts/train_model.py`` may fit a small
   Random Forest to those judgements for study. Nothing here produces the
   AI-Assisted Assessment shown on reports - that comes from the
   **pre-trained CLIP model** in :mod:`app.ai.clip_analyzer`.

Honest scope
------------
This is a small, genuinely trained scikit-learn model. It learns the
relationship between 12 measurable visual features and a quality score
that a human assigned. It is NOT a large pre-trained neural network, and it
does not "understand" design the way a person does. It was superseded by
the zero-shot CLIP approach, which needs no labelled data at all.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

from app.cv.feature_extraction import FEATURE_NAMES, to_vector

# Where the trained artefacts live.
MODEL_PATH = os.path.join(os.path.dirname(__file__), "aesthetic_model.pkl")
METRICS_PATH = os.path.join(os.path.dirname(__file__), "aesthetic_model_metrics.json")

# Labels are stored as 0-10 quality scores; the UI shows 0-100.
SCORE_SCALE = 10.0

# Refuse to train on less than this — too little data produces a model
# that guesses, and pretending otherwise would be dishonest.
MIN_TOTAL_SAMPLES = 8
MIN_PER_CLASS = 2


@dataclass
class ModelInfo:
    """Everything a reader needs to judge whether the model is trustworthy."""

    trained_at: str
    n_samples: int
    n_good: int
    n_weak: int
    n_augmented: int
    mae: float                 # cross-validated mean absolute error (0-100 scale)
    discrimination: float      # share of good/weak pairs ranked correctly
    feature_importance: dict   # {feature name: 0-1 share}


class TrainingDataError(ValueError):
    """Raised when there is not enough or not varied enough labelled data."""


# =========================================================
# TRAINING
# =========================================================

def train_model(feature_rows: list[dict],
                targets: list[float],
                n_augmented: int = 0) -> tuple[RandomForestRegressor, ModelInfo]:
    """
    Train the aesthetic model.

    Args:
        feature_rows: one dict of the 12 features per labelled design.
        targets: quality score per design, 0-10 (from LABEL_SCORES).
        n_augmented: how many extra synthetic copies were included.

    Returns:
        (fitted model, ModelInfo)

    Raises:
        TrainingDataError: if the data is too small or too one-sided.
    """
    if len(feature_rows) != len(targets):
        raise TrainingDataError(
            f"Feature rows ({len(feature_rows)}) and targets "
            f"({len(targets)}) must line up."
        )

    y = np.asarray(targets, dtype=np.float64)
    X = np.vstack([to_vector(row) for row in feature_rows])

    n_good = int((y >= 5.0).sum())
    n_weak = int((y < 5.0).sum())

    if len(y) < MIN_TOTAL_SAMPLES:
        raise TrainingDataError(
            f"Only {len(y)} labelled design(s). At least {MIN_TOTAL_SAMPLES} "
            f"are needed before training is meaningful."
        )
    if n_good < MIN_PER_CLASS or n_weak < MIN_PER_CLASS:
        raise TrainingDataError(
            f"Need at least {MIN_PER_CLASS} good AND {MIN_PER_CLASS} weak "
            f"examples (currently {n_good} good / {n_weak} weak). A model "
            f"trained on one side only cannot tell the difference."
        )

    # Random forest: robust on small tabular data, and it reports which
    # features drove each prediction so the result stays explainable.
    model = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=2,       # guards against memorising 1 example
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)

    mae, discrimination = _cross_validated_scores(X, y)

    importances = {
        name: round(float(imp), 4)
        for name, imp in zip(FEATURE_NAMES, model.feature_importances_)
    }

    info = ModelInfo(
        trained_at=datetime.utcnow().isoformat(timespec="seconds"),
        n_samples=len(y),
        n_good=n_good,
        n_weak=n_weak,
        n_augmented=n_augmented,
        mae=round(mae, 2),
        discrimination=round(discrimination, 4),
        feature_importance=dict(
            sorted(importances.items(), key=lambda kv: kv[1], reverse=True)
        ),
    )
    return model, info


def _cross_validated_scores(X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """
    Honest internal evaluation.

    Every design is predicted by a model that did NOT train on it, so the
    figures below describe unseen designs rather than memorised ones.

    Leave-one-out would be ideal for tiny datasets but refits the model
    once per design, which is far too slow as the set grows. So: leave
    one-out only while the set is very small, then 5-fold after that.

    Returns:
        (mean absolute error in 0-100 points, fraction of good/weak pairs
         the model ranks in the right order)
    """
    n = len(y)
    predictions = np.zeros(n, dtype=np.float64)

    n_splits = n if n <= 8 else 5
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for train_idx, test_idx in splitter.split(X):
        fold = RandomForestRegressor(
            n_estimators=120,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        )
        fold.fit(X[train_idx], y[train_idx])
        predictions[test_idx] = fold.predict(X[test_idx])

    mae = float(np.mean(np.abs(predictions - y))) * SCORE_SCALE

    # Ranking quality: does a "good" design tend to outscore a "weak" one?
    good_idx = np.where(y >= 5.0)[0]
    weak_idx = np.where(y < 5.0)[0]
    if len(good_idx) and len(weak_idx):
        pairs = [(g, w) for g in good_idx for w in weak_idx]
        correct = sum(predictions[g] > predictions[w] for g, w in pairs)
        discrimination = correct / len(pairs)
    else:
        discrimination = 0.0

    return mae, discrimination


# =========================================================
# PERSISTENCE
# =========================================================

def save_model(model: RandomForestRegressor, info: ModelInfo) -> None:
    """Write the fitted model and its metrics next to each other."""
    joblib.dump(model, MODEL_PATH)
    with open(METRICS_PATH, "w", encoding="utf-8") as fh:
        json.dump(asdict(info), fh, indent=2)


def load_model() -> tuple[RandomForestRegressor | None, ModelInfo | None]:
    """
    Load the trained model, or (None, None) when none exists yet.

    A missing model is a normal state, not an error: the app keeps working
    on rule-based scores alone.
    """
    if not os.path.isfile(MODEL_PATH):
        return None, None
    try:
        model = joblib.load(MODEL_PATH)
    except Exception:
        return None, None

    info = None
    if os.path.isfile(METRICS_PATH):
        try:
            with open(METRICS_PATH, encoding="utf-8") as fh:
                info = ModelInfo(**json.load(fh))
        except Exception:
            info = None
    return model, info


def is_available() -> bool:
    """True when a trained model is on disk and loadable."""
    model, _ = load_model()
    return model is not None


def clear_model() -> bool:
    """Delete the trained artefacts (used when retraining from scratch)."""
    removed = False
    for path in (MODEL_PATH, METRICS_PATH):
        if os.path.isfile(path):
            os.remove(path)
            removed = True
    return removed


# =========================================================
# PREDICTION
# =========================================================

def predict_score(model: RandomForestRegressor, features: dict) -> float:
    """
    Predict the aesthetic quality of one design.

    Returns:
        A 0-100 quality estimate.
    """
    vector = to_vector(features).reshape(1, -1)
    raw = float(model.predict(vector)[0])
    return float(np.clip(raw * SCORE_SCALE, 0.0, 100.0))


def top_factors(model: RandomForestRegressor,
                features: dict,
                limit: int = 3) -> list[dict]:
    """
    The features that moved this prediction most.

    Uses each feature's share of the model, scaled by how far the design
    sits from the middle of that feature's range. This keeps the claim
    modest: these are the model's main inputs, not a causal explanation.
    """
    vector = to_vector(features).reshape(1, -1)
    importances = model.feature_importances_

    scored = []
    for name, imp, value in zip(FEATURE_NAMES, importances, vector[0]):
        deviation = abs(value - 0.5) * 2.0        # 0 at the midpoint, 1 at an edge
        scored.append({
            "feature": name,
            "importance": round(float(imp), 4),
            "value": round(float(value), 3),
            "weight": round(float(imp) * deviation, 4),
        })
    scored.sort(key=lambda item: item["weight"], reverse=True)
    return scored[:limit]

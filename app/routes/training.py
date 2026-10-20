"""Training data collection (AI Phase 11, step 2).

Admin-only pages that let a human judge designs as "good" or "weak".
Those judgements are the training examples the aesthetic model learns from.

Pipeline position:
    IMAGE -> PRE-PROCESSING -> CV FEATURES -> (labelled by a human here)
"""
from __future__ import annotations

import json
import os

from flask import (Blueprint, abort, flash, redirect, render_template,
                   request, url_for)

from app.models import database
from app.models.database import LABEL_SCORES
from app.routes.auth import admin_required, login_required
from app.cv.preprocessing import prepare
from app.cv.feature_extraction import extract_features


training_bp = Blueprint("training", __name__, url_prefix="/admin/label")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
UPLOADS = os.path.join(PROJECT_ROOT, "static", "uploads")


def _listed_designs() -> list[str]:
    """Relative paths of every stored design, sorted for a stable order."""
    if not os.path.isdir(UPLOADS):
        return []
    allowed = {"png", "jpg", "jpeg", "webp", "bmp", "gif"}
    out = []
    for name in sorted(os.listdir(UPLOADS)):
        if name.rsplit(".", 1)[-1].lower() in allowed:
            out.append(f"static/uploads/{name}")
    return out


def _counts(done: int, total: int) -> dict:
    return database.get_label_counts() | {"done": done, "total": total}


@training_bp.route("/")
@login_required
@admin_required
def index():
    designs = _listed_designs()
    labelled = database.get_labelled_paths()
    pending = [p for p in designs if p not in labelled]
    counts = _counts(len(labelled & set(designs)), len(designs))
    return render_template("admin/label.html", pending=pending, counts=counts)


@training_bp.route("/save", methods=["POST"])
@login_required
@admin_required
def save():
    image_path = request.form.get("image_path", "").strip()
    label = request.form.get("label", "").strip()

    if label not in LABEL_SCORES:
        flash("Unknown label. Please choose Good or Weak.")
        return redirect(url_for("training.index"))
    if not image_path.startswith("static/uploads/"):
        flash("Invalid design reference.")
        return redirect(url_for("training.index"))

    abs_path = os.path.join(PROJECT_ROOT, image_path)
    if not os.path.isfile(abs_path):
        flash("That design file no longer exists.")
        return redirect(url_for("training.index"))

    # Cache the 12 features now so training later is instant.
    data = prepare(abs_path)
    features = extract_features(data["rgb"])
    database.save_label(
        image_path=image_path,
        label=label,
        score=LABEL_SCORES[label],
        features_json=json.dumps(features),
    )
    return redirect(url_for("training.index"))


@training_bp.route("/undo", methods=["POST"])
@login_required
@admin_required
def undo():
    """Take back the most recent label so a wrong click is easy to fix."""
    records = database.get_labelled_records()
    if not records:
        flash("Nothing to undo.")
        return redirect(url_for("training.index"))
    last = records[-1]
    database.delete_label(last["image_path"])
    flash("Removed the last label — it is back in the queue.")
    return redirect(url_for("training.index"))


@training_bp.route("/reset", methods=["POST"])
@login_required
@admin_required
def reset():
    for path in list(database.get_labelled_paths()):
        database.delete_label(path)
    flash("All labels cleared.")
    return redirect(url_for("training.index"))


@training_bp.route("/export")
@login_required
@admin_required
def export():
    """Download the training set as JSON (for the report / retraining)."""
    from flask import jsonify
    return jsonify({"examples": database.get_labelled_records()})

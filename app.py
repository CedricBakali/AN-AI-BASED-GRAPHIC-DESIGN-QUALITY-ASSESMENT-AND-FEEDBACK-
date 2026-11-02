"""
DesignScore - Flask web application.

Upload a finished graphic design (poster, advert, social post) and receive an
instant, rule-based quality review across six core design principles.
"""
from __future__ import annotations

import os
import uuid

from flask import (Flask, abort, flash, jsonify, redirect, render_template, request,
                   session, url_for)
from werkzeug.utils import secure_filename

from app.models import database
from app.models.user import get_user_by_id
from app.services.analysis import analyze_image
from app.services.feedback import build_feedback
from app.ai import clip_analyzer, model_loader
from app.routes.auth import auth_bp, login_required
from app.routes.dashboard import dashboard_bp
from app.routes.admin import admin_bp
from app.routes.training import training_bp


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
ALLOWED_EXT = {"png", "jpg", "jpeg", "webp", "bmp", "gif"}
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB


app = Flask(__name__, template_folder="app/templates")

# NOTE: app/routes/analysis.py also defines POST /analyze but is intentionally
# NOT registered here to avoid "View function mapping is overwriting" collision
# with the /analyze route defined below. Pick one pattern long-term.

app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
app.secret_key = os.environ.get("DESIGNSCORE_SECRET", "dev-secret-change-me")

os.makedirs(UPLOAD_DIR, exist_ok=True)
database.init_db()
app.register_blueprint(auth_bp)
app.register_blueprint(dashboard_bp)
app.register_blueprint(admin_bp)
app.register_blueprint(training_bp)

# Load the pre-trained CLIP model ONCE, in the background, so the first
# user upload is not the one paying the ~30 s cold-import cost. Nothing is
# trained; this is inference setup only. Disable with
# DESIGNSCORE_CLIP_WARMUP=0 (the test suite does exactly that).
if os.environ.get("DESIGNSCORE_CLIP_WARMUP", "1").strip().lower() not in {
        "0", "false", "no", "off"}:
    model_loader.start_background_warmup()


def _allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def _ai_factors(ai: dict) -> list[dict]:
    """Turn the CLIP readings into the factor list the report renders.

    Keeps the existing {name, value, reading} shape used by result.html.
    ``value`` is always the probability that the POSITIVE description
    matched (0-1). These are general impressions - never presented as
    measurements of a specific design principle.
    """
    factors: list[dict] = []

    probability = ai.get("positive_probability")
    if probability is None and ai.get("score") is not None:
        probability = float(ai["score"]) / 100.0
    if probability is not None:
        p = float(probability)
        factors.append({
            "name": "Overall design quality",
            "value": round(p, 4),
            "reading": (f"positive lean ({round(p * 100)}%)"
                        if p >= 0.5 else
                        f"negative lean ({round((1 - p) * 100)}%)"),
        })

    for impression in ai.get("impressions") or []:
        p = float(impression.get("positive_probability") or 0.0)
        factors.append({
            "name": impression.get("label") or "Impression",
            "value": round(p, 4),
            "reading": (f"positive lean ({round(p * 100)}%)"
                        if p >= 0.5 else
                        f"negative lean ({round((1 - p) * 100)}%)"),
        })
    return factors


@app.route("/")
def index():
    # Send logged-in users to their workspace, others to login.
    if "user_id" in session:
        return redirect(url_for("dashboard.dashboard"))
    return redirect(url_for("auth.login"))


@app.route("/analyze", methods=["POST"])
@login_required
def analyze():
    # The JS uploader sends X-Requested-With so it can show a progress bar and
    # handle errors inline. A plain browser form still gets the normal
    # redirect + flash behaviour.
    wants_json = request.headers.get("X-Requested-With") == "XMLHttpRequest"

    def fail(message: str):
        if wants_json:
            return jsonify({"ok": False, "message": message}), 400
        flash(message)
        return redirect(url_for("dashboard.dashboard"))

    if "image" not in request.files:
        return fail("No file was uploaded.")
    file = request.files["image"]
    if not file or file.filename == "":
        return fail("Please choose an image to analyze.")
    if not _allowed(file.filename):
        return fail("Unsupported file type. Use PNG, JPG, WEBP, BMP or GIF.")

    safe = secure_filename(file.filename)
    unique = f"{uuid.uuid4().hex}_{safe}"
    # Forward slashes: this string is stored in DB and served via /static/<...>
    rel_path = f"static/uploads/{unique}"
    abs_path = os.path.join(UPLOAD_DIR, unique)
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file.save(abs_path)

    try:
        result = analyze_image(abs_path)
    except Exception as exc:  # noqa: BLE001 - surface a friendly message
        return fail(f"Could not analyze that image: {exc}")

    # Pre-trained CLIP estimate (ZERO-SHOT INFERENCE ONLY - nothing is
    # trained here). Strictly optional: a missing, slow or broken model must
    # never stop the rule-based report from being produced. When it cannot
    # run, ai_score stays NULL and the report says the AI assessment is
    # unavailable.
    ai_score, ai_factors, ai_model = None, None, None
    try:
        ai = clip_analyzer.analyze_with_ai(abs_path)
    except Exception:  # noqa: BLE001 - degrade to rule-based only
        ai = None
    if ai and ai.get("available") and ai.get("score") is not None:
        ai_score = int(ai["score"])
        ai_model = ai.get("model") or clip_analyzer.MODEL_ID
        ai_factors = _ai_factors(ai)

    analysis_id = database.save_analysis(
        session["user_id"], safe, rel_path, result,
        ai_score=ai_score, ai_factors=ai_factors, ai_model=ai_model,
    )
    if wants_json:
        return jsonify({
            "ok": True,
            "url": url_for("result", analysis_id=analysis_id),
        })
    return redirect(url_for("result", analysis_id=analysis_id))


def _ai_panel(record: dict) -> dict:
    """Everything the 'AI-Assisted Assessment' panel needs.

    Built purely from what was stored at analysis time, so a report stays
    stable even if the model is later replaced. Never raises.
    """
    score = record.get("ai_score")
    factors = record.get("ai_factors") or []
    model_id = record.get("ai_model") or clip_analyzer.MODEL_ID
    base = {
        "model": model_id,
        "label": clip_analyzer.MODEL_LABEL,
        "factors": factors if isinstance(factors, list) else [],
    }

    if score is None:
        base.update({
            "available": False,
            "status": "Unavailable",
            "score": None,
            "explanation": "",
            "probability": None,
            "gap": None,
        })
        return base

    probability = None
    if base["factors"] and isinstance(base["factors"][0], dict):
        probability = base["factors"][0].get("value")
    if probability is None:
        probability = float(score) / 100.0

    nearer = "first" if probability >= 0.5 else "second"
    # NOTE: this string is rendered through Jinja's auto-escaping, so use
    # real Unicode characters - HTML entities here would be double-escaped
    # and shown to the user as literal "&mdash;".
    explanation = (
        'The pre-trained CLIP model compared this design with two opposing '
        'descriptions \u2014 "' + clip_analyzer.SCORE_POSITIVE +
        '" versus "' + clip_analyzer.SCORE_NEGATIVE + '" \u2014 and '
        'aligned more closely with the ' + nearer + ' (' +
        str(round(probability * 100)) + '% versus ' +
        str(round((1 - probability) * 100)) + '%). This is a zero-shot '
        'AI assessment: the model was pre-trained by OpenAI and nothing '
        'was trained or fine-tuned on this project\u2019s data.'
    )

    base.update({
        "available": True,
        "status": "Available",
        "score": int(score),
        "explanation": explanation,
        "probability": round(float(probability), 3),
        "gap": int(score) - int(record.get("overall") or 0),
    })
    return base


@app.route("/result/<int:analysis_id>")
@login_required
def result(analysis_id: int):
    record = database.get_analysis(analysis_id, session["user_id"])
    if record is None:
        abort(404)
    fb = build_feedback(record["categories"])
    return render_template("analysis/result.html", r=record, fb=fb,
                           ai=_ai_panel(record))


@app.route("/history")
@login_required
def history():
    user_id = session["user_id"]
    records = database.get_history(user_id, limit=30)
    stats = database.get_user_stats(user_id)
    user = get_user_by_id(user_id)
    return render_template(
        "history/history.html",
        records=records,
        stats=stats,
        user=user,
    )


@app.errorhandler(413)
def too_large(_e):
    message = "That file is too large (max 10 MB)."
    if request.headers.get("X-Requested-With") == "XMLHttpRequest":
        return jsonify({"ok": False, "message": message}), 413
    flash(message)
    return redirect(url_for("dashboard.dashboard"))


if __name__ == "__main__":
    # Bind to localhost only.
    app.run(host="127.0.0.1", port=5000, debug=True)

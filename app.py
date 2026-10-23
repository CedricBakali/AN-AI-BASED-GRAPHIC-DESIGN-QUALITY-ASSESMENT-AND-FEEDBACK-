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


def _allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


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

    analysis_id = database.save_analysis(session["user_id"], safe, rel_path, result)
    if wants_json:
        return jsonify({
            "ok": True,
            "url": url_for("result", analysis_id=analysis_id),
        })
    return redirect(url_for("result", analysis_id=analysis_id))


@app.route("/result/<int:analysis_id>")
@login_required
def result(analysis_id: int):
    record = database.get_analysis(analysis_id, session["user_id"])
    if record is None:
        abort(404)
    fb = build_feedback(record["categories"])
    return render_template("analysis/result.html", r=record, fb=fb)


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

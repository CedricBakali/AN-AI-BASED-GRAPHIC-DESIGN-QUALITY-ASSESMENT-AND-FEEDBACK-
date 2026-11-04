"""
DesignScore - Flask web application.

Upload a finished graphic design (poster, advert, social post) and receive an
instant, rule-based quality review across six core design principles.
"""
from __future__ import annotations

import base64
import datetime
import os
import secrets
import uuid

from flask import (Flask, Response, abort, flash, jsonify, redirect,
                   render_template, request, session, url_for)
from werkzeug.utils import secure_filename

from app.models import database
from app.models.user import get_user_by_id
from app.services.analysis import analyze_image
from app.services.feedback import build_feedback
from app.ai import clip_analyzer, model_loader, nima_analyzer
from app.routes.auth import auth_bp, login_required
from app.routes.dashboard import dashboard_bp
from app.routes.admin import admin_bp
from app.routes.training import training_bp
from config import debug_enabled, resolve_secret_key


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
ALLOWED_EXT = {"png", "jpg", "jpeg", "webp", "bmp", "gif"}
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB


app = Flask(__name__, template_folder="app/templates")

# NOTE: app/routes/analysis.py also defines POST /analyze but is intentionally
# NOT registered here to avoid "View function mapping is overwriting" collision
# with the /analyze route defined below. Pick one pattern long-term.

app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# Session-signing key. The environment variable wins; otherwise a random
# key is generated once and kept in a gitignored file. The previous
# hard-coded fallback was a published constant that would have let anyone
# forge a session cookie, so it is gone - see config.resolve_secret_key().
app.secret_key = resolve_secret_key()

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


# ------------------------------------------------------------------
# Uploaded designs are PRIVATE.
#
# Flask serves everything under /static/ with no authentication, so until
# now any uploaded design could be fetched by anyone who could reach this
# server - no login, no ownership check, just the file name. Upload names
# are uuid4-based and therefore unguessable, but "hard to guess" is not a
# permission model.
#
# Every request for an uploaded file must now prove either that the caller
# owns it or that they are an administrator (the labelling page has to show
# every design). Anything else is a 404 rather than a 403, so the existence
# of other people's files is not confirmed either.
#
# Doing the check here means no template has to change: <img> tags keep
# pointing at the same URLs and simply stop working for strangers.
# ------------------------------------------------------------------
UPLOAD_URL_PREFIX = "/static/uploads/"


def _image_owner(image_path: str) -> int | None:
    """The user id whose analysis used this upload, if any."""
    try:
        with database.get_connection() as conn:
            row = conn.execute(
                "SELECT user_id FROM analyses WHERE image_path = ? "
                "ORDER BY id DESC LIMIT 1",
                (image_path,),
            ).fetchone()
    except Exception:  # noqa: BLE001 - a failed lookup must deny, not allow
        return None
    if row is None or row["user_id"] is None:
        return None
    return int(row["user_id"])


@app.before_request
def _protect_uploaded_designs():
    """Refuse direct static access to uploads unless the caller owns it."""
    path = request.path
    if not path.startswith(UPLOAD_URL_PREFIX):
        return None

    # Uploads are flat file names only. Anything nested, absolute or
    # carrying a parent reference is rejected before it can be used to walk
    # out of the folder (request.path is already URL-decoded).
    name = path[len(UPLOAD_URL_PREFIX):]
    if (not name or "/" in name or "\\" in name or ".." in name
            or "\x00" in name):
        abort(404)

    user_id = session.get("user_id")
    if not user_id:
        abort(404)

    user = get_user_by_id(user_id)
    if user is None or not user["is_active"]:
        abort(404)
    if user["role"] == "admin":
        return None      # administrators review every submitted design

    if _image_owner(UPLOAD_URL_PREFIX.lstrip("/") + name) != int(user_id):
        abort(404)
    return None


# ------------------------------------------------------------------
# Cross-Site Request Forgery
#
# Every state-changing route (login, register, upload, password change,
# labelling) used to accept a bare POST: a page on another site could
# submit one using the victim's own session cookie, and the request would
# be indistinguishable from the real thing.
#
# Each session now gets one random token. Templates render it into their
# forms with {{ csrf_token() }}, and every POST has to echo it back -
# either as a hidden field or as an X-CSRFToken header. A page on another
# domain can read its own forms, never this session's token, so its
# submissions are refused.
#
# The token is compared with secrets.compare_digest so that response
# timing cannot be used to guess it.
# ------------------------------------------------------------------
CSRF_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS", "TRACE"})
CSRF_FIELD = "csrf_token"


def _csrf_token() -> str:
    """The one random token for this session, created on first use."""
    token = session.get("_csrf")
    if not token:
        token = secrets.token_urlsafe(32)
        session["_csrf"] = token
    return token


@app.context_processor
def _inject_csrf_token():
    """Expose {{ csrf_token() }} to every template."""
    return {"csrf_token": _csrf_token}


def _csrf_rejected():
    """Refuse a POST whose token is missing, stale or simply not ours."""
    if request.headers.get("X-Requested-With") == "XMLHttpRequest":
        # The uploader shows this text inline instead of reloading.
        return jsonify(ok=False,
                       message="Your session expired. Reload the page and "
                               "try again."), 400
    flash("Your session expired. Please try that again.")
    return redirect(request.referrer or url_for("dashboard.dashboard"))


@app.before_request
def _verify_csrf():
    """Reject any state-changing request that cannot prove its origin."""
    if request.method in CSRF_SAFE_METHODS:
        return None

    # The two test harnesses drive the app with test_client() and never
    # render a form, so enforcement is lifted whenever TESTING is set.
    # Nothing outside a test ever sets TESTING, so this cannot weaken a
    # live deployment - and the dedicated CSRF tests turn it back on.
    if app.config.get("TESTING") or not app.config.get("CSRF_ENABLED", True):
        return None

    expected = session.get("_csrf")
    if not expected:
        return _csrf_rejected()

    supplied = (request.form.get(CSRF_FIELD)
                or request.headers.get("X-CSRFToken")
                or "")
    if not supplied or not secrets.compare_digest(str(supplied), str(expected)):
        return _csrf_rejected()
    return None


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


# ---------------------------------------------------------------------------
# Plain-English wording for the result page and the full report.
#
# Nothing here changes a score. These helpers only translate numbers that
# were already measured into words a beginner designer can read, so the
# user-facing pages never have to show raw prompt-pair percentages.
# ---------------------------------------------------------------------------

#: Shared bands for the AI opinion. Fixed thresholds, reused for the AI
#: score and for every impression, so a number always maps to the same
#: word. No invented precision.
AI_BANDS = (
    (85, "Excellent"),
    (70, "Very good"),
    (55, "Good"),
    (45, "Fair"),
)

#: Reading used for any score below the lowest band.
BOTTOM_WORD = "Needs improvement"

#: Same bottom band, reworded where a different phrase reads better.
BOTTOM_WORD_OVERRIDES = {"Attention-grabbing": "Could be improved"}

#: Friendly names for the stored AI impressions.
IMPRESSION_NAMES = {
    "Overall design quality": "Overall visual impression",
    "Attractiveness": "Visual appeal",
    "Professionalism": "Professional appearance",
    "Attention-grabbing": "Visual impact",
}


def ai_band(score) -> str:
    """Turn a 0-100 AI score into a short plain-English reading."""
    if score is None:
        return "Unavailable"
    value = int(score)
    for threshold, label in AI_BANDS:
        if value >= threshold:
            return label
    return BOTTOM_WORD


def _impression_row(factor: dict) -> dict:
    """Translate one stored AI reading into friendly words.

    Returns the same {name, value, shape} shape the report already uses,
    but with a human label and a band instead of "positive/negative lean".
    """
    name = str(factor.get("name") or "Impression")
    probability = float(factor.get("value") or 0.0)
    reading = ai_band(int(round(probability * 100)))
    if reading == BOTTOM_WORD:
        reading = BOTTOM_WORD_OVERRIDES.get(name, BOTTOM_WORD)
    return {
        "name": IMPRESSION_NAMES.get(name, name),
        "value": round(probability, 4),
        "reading": reading,
    }


def _ai_feedback(rows: list[dict]) -> str:
    """One short, honest sentence built from the translated impressions."""
    if not rows:
        return ""
    best = max(rows, key=lambda row: row["value"])
    worst = min(rows, key=lambda row: row["value"])
    if best is worst:
        return (f"Your {best['name'].lower()} reads as "
                f"{best['reading'].lower()}.")
    line = (f"Your {best['name'].lower()} was the strongest point "
            f"({best['reading'].lower()}).")
    if worst["reading"] in {BOTTOM_WORD, "Could be improved", "Fair"}:
        line += f" Your {worst['name'].lower()} needs the most work."
    return line


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

    # SECOND, independent AI opinion (NIMA) - a separate model on its own
    # native 0-10 scale. Same fail-soft contract as CLIP: a missing model,
    # a cold network or a slow CPU must never stop the report being saved.
    # Its result is stored in its own columns and is never combined with
    # ai_score (0-100) or with the design-quality score.
    nima_score, nima_model = None, None
    try:
        nima = nima_analyzer.analyze_with_nima(abs_path)
    except Exception:  # noqa: BLE001 - a second opinion is always optional
        nima = None
    if nima and nima.get("available") and nima.get("score") is not None:
        nima_score = float(nima["score"])
        nima_model = nima.get("model") or nima_analyzer.MODEL_ID

    analysis_id = database.save_analysis(
        session["user_id"], safe, rel_path, result,
        ai_score=ai_score, ai_factors=ai_factors, ai_model=ai_model,
        nima_score=nima_score, nima_model=nima_model,
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
            "band": "",
            "simple": [],
            "feedback": "",
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

    simple = [_impression_row(f) for f in base["factors"]
              if isinstance(f, dict)]

    base.update({
        "available": True,
        "status": "Available",
        "score": int(score),
        "band": ai_band(int(score)),
        "simple": simple,
        "feedback": _ai_feedback(simple),
        "explanation": explanation,
        "probability": round(float(probability), 3),
        "gap": int(score) - int(record.get("overall") or 0),
    })
    return base


# ---------------------------------------------------------------------------
# Full assessment report: viewing it and downloading it.
# ---------------------------------------------------------------------------

def _date_label(value) -> str:
    """Turn the stored timestamp into a date a person can read.

    '2026-11-02T07:41:07' -> '2 November 2026, 07:41'
    """
    text = str(value or "")
    try:
        stamp = datetime.datetime.fromisoformat(text)
    except ValueError:
        return text[:10]
    return (f"{stamp.day} {stamp.strftime('%B %Y')}, "
            f"{stamp.strftime('%H:%M')}")


def _report_css() -> str:
    """Inlined so a downloaded report still looks right with no server."""
    path = os.path.join(BASE_DIR, "static", "css", "report.css")
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    except OSError:
        return ""


def _preview_src(record: dict, download: bool) -> str:
    """The design preview.

    Viewing uses the normal static URL. A download embeds the image as a
    data URI so the saved file still shows the design on its own.
    """
    rel = record.get("image_path") or ""
    if not download:
        return "/" + rel
    if not rel:
        return ""
    # Only ever read the file name, and only from inside the upload folder.
    name = rel.replace("\\", "/").rsplit("/", 1)[-1]
    root = os.path.realpath(UPLOAD_DIR)
    path = os.path.realpath(os.path.join(root, name))
    if not path.startswith(root + os.sep):
        return ""
    try:
        with open(path, "rb") as fh:
            data = fh.read()
    except OSError:
        return ""
    ext = (name.rsplit(".", 1)[-1] or "png").lower()
    mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp",
            "bmp": "bmp", "gif": "gif"}.get(ext, "png")
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:image/{mime};base64,{encoded}"


#: Bands for NIMA's own 0-10 scale, in the same spirit as AI_BANDS:
#: fixed thresholds, so the same number always maps to the same words.
#: These only turn an already measured number into plain English - they
#: change no score. 5.5 is the midpoint of the 1..10 scale NIMA predicts,
#: so the top band starts well above it.
#:
#: Each band carries the message the designer reads first (feedback about
#: THIS image) and a concrete next step. NIMA sees the picture as a whole,
#: so both are written as a first-impression reading, never as a verdict
#: on a specific design principle.
NIMA_BANDS = (
    (7.5, "Strong",
     "At a glance your design lands well - this is the kind of image "
     "viewers tend to rate highly before they read a single word.",
     "Keep the overall colour, light and composition as they are; they "
     "are carrying this design."),
    (6.0, "Good",
     "Your design looks good at a glance - comfortably above what NIMA "
     "would call an average image.",
     "It is close to the top of its scale. A little more polish in "
     "contrast and spacing would push it into the clearly strong range."),
    (4.5, "Average",
     "Your design reads as average - nothing is wrong with it, but "
     "nothing stops a viewer either.",
     "Give the eye somewhere to land first: one dominant element, and "
     "clearer contrast between it and everything else."),
    (3.0, "Below average",
     "At a glance your design sits below average - as an image it does "
     "not hold attention.",
     "Strip it back. Fewer competing elements and a single focal point "
     "will lift this kind of impression fastest."),
)

#: The band used for anything under the lowest threshold above.
NIMA_BOTTOM = (
    "Weak",
    "As an image your design gives a weak first impression - it does "
    "not land with a viewer.",
    "Start with the fundamentals: one focal point, a limited palette and "
    "strong contrast between the main elements and the background.",
)


def _nima_panel(record: dict) -> dict:
    """The second AI opinion, deliberately kept apart from ``ai``.

    NIMA returns a predicted rating on its own 0-10 scale. That number is
    never compared with, added to, or averaged against the 0-100 scores,
    and nothing here changes any score - it only turns an already measured
    number into words. Never raises.
    """
    score = record.get("nima_score")
    model_id = record.get("nima_model") or nima_analyzer.MODEL_ID

    panel = {
        "model": model_id,
        "label": nima_analyzer.MODEL_LABEL,
        "scale": "10",
        "score": None if score is None else round(float(score), 1),
        "available": score is not None,
        "status": "Unavailable" if score is None else "Available",
        "band": "",
        "feedback": "",
        "advice": "",
        "reading": "",
        "explanation": "",
    }
    if score is None:
        return panel

    value = float(score)

    # What the designer reads first: a short verdict on this image plus
    # one next step, picked from the fixed table above.
    band, feedback, advice = NIMA_BOTTOM
    for threshold, word, message, recommendation in NIMA_BANDS:
        if value >= threshold:
            band, feedback, advice = word, message, recommendation
            break

    # Position on NIMA's own scale - a factual statement about where the
    # number sits, not a quality verdict invented here. 5.5 is the midpoint
    # of the 1..10 scale NIMA was trained to predict.
    if value > 5.5:
        place = "above the middle"
    elif value < 5.5:
        place = "below the middle"
    else:
        place = "right at the middle"

    panel.update({
        "band": band,
        "feedback": feedback,
        "advice": advice,
        "reading": (f"NIMA predicts an average rating of {value:.1f} out of "
                    f"10, {place} of its scale."),
        "explanation": (
            "NIMA is a different network that predicts how people rate an "
            "image on a 1-10 scale. It looks at the picture only: it does "
            "not use the eight design principles and it shares no "
            "calculation with either score above. It was trained on "
            "photographs, so read it as a broad visual impression rather "
            "than a verdict on graphic design."
        ),
    })
    return panel


def _report_context(analysis_id: int, download: bool) -> dict:
    """Everything both report pages need. Never invents a value."""
    record = database.get_analysis(analysis_id, session["user_id"])
    if record is None:
        abort(404)
    return {
        "r": record,
        "fb": build_feedback(record["categories"]),
        "ai": _ai_panel(record),
        "nima": _nima_panel(record),
        "download": download,
        "css": _report_css(),
        "preview_src": _preview_src(record, download),
        "when": _date_label(record.get("created_at")),
    }


@app.route("/report/<int:analysis_id>")
@login_required
def report(analysis_id: int):
    """The complete assessment, as a readable document."""
    return render_template("analysis/report.html",
                           **_report_context(analysis_id, download=False))


@app.route("/report/<int:analysis_id>/download")
@login_required
def report_download(analysis_id: int):
    """The same document as a self-contained file the user can keep."""
    ctx = _report_context(analysis_id, download=True)
    body = render_template("analysis/report.html", **ctx)
    filename = f"DesignScore-Report-{analysis_id}.html"
    return Response(body, mimetype="text/html", headers={
        "Content-Disposition": f'attachment; filename="{filename}"',
    })


@app.route("/result/<int:analysis_id>")
@login_required
def result(analysis_id: int):
    record = database.get_analysis(analysis_id, session["user_id"])
    if record is None:
        abort(404)
    fb = build_feedback(record["categories"])
    return render_template("analysis/result.html", r=record, fb=fb,
                           ai=_ai_panel(record),
                           nima=_nima_panel(record),
                           when=_date_label(record.get("created_at")))


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
    # Bind to localhost only. debug_enabled() keeps the interactive
    # Werkzeug debugger off unless FLASK_DEBUG=1 is set explicitly.
    app.run(host="127.0.0.1", port=5000, debug=debug_enabled())

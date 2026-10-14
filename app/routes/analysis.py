"""Upload helpers for graphic designs.

The canonical POST /analyze route lives in app.py (tested working:
UUID filenames, root static/uploads, per-user save).
This module keeps the ``analysis`` blueprint object for target-architecture
compatibility but defines NO routes, so there is no duplicate /analyze.
"""
from __future__ import annotations

from flask import Blueprint

analysis_bp = Blueprint("analysis", __name__)

try:
    from config import Config
    ALLOWED_EXTENSIONS = set(Config.ALLOWED_EXTENSIONS)
    UPLOAD_FOLDER = Config.UPLOAD_FOLDER
except Exception:  # config not importable in some contexts; safe fallback
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp", "gif"}
    UPLOAD_FOLDER = "static/uploads"


def allowed_file(filename: str) -> bool:
    """Check whether the uploaded file has an allowed image extension."""
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )

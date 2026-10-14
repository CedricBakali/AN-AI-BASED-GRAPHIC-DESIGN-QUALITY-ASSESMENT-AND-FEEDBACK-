"""Central Flask configuration for DesignScore.

Use:  app.config.from_object("config.Config")
"""
from __future__ import annotations

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class Config:
    SECRET_KEY = os.environ.get("DESIGNSCORE_SECRET", "dev-secret-change-me")
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp", "gif"}
    DATABASE_PATH = os.path.join(BASE_DIR, "app", "models", "designscore.db")

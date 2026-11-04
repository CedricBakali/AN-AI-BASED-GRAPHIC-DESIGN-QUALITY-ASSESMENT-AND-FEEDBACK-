"""Central Flask configuration for DesignScore.

Use:  app.config.from_object("config.Config")
"""
from __future__ import annotations

import os
import secrets

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

#: Where a generated session key is kept when DESIGNSCORE_SECRET is unset.
#: Gitignored (see .gitignore) - a session-signing key must never be
#: committed, published or shipped in a repository.
SECRET_KEY_FILE = os.path.join(BASE_DIR, ".designscore_secret")


def resolve_secret_key() -> str:
    """The Flask session-signing key, resolved safely.

    Priority:

    1. ``DESIGNSCORE_SECRET`` in the environment. It always wins, so a
       redeploy with the same variable keeps everyone logged in.
    2. The random key persisted in ``.designscore_secret``, generated once
       on the first run and reused after that - sessions therefore survive
       a restart without the key being a known constant.
    3. Otherwise generate one and persist it. If the checkout is read-only
       the key still exists for this process, it simply is not saved.

    The previous hard-coded fallback has been removed. A published default
    is a real weakness: anyone who knows it can forge a signed session
    cookie and impersonate any user, including an administrator.
    """
    env = os.environ.get("DESIGNSCORE_SECRET", "").strip()
    if env:
        return env

    try:
        with open(SECRET_KEY_FILE, encoding="utf-8") as fh:
            stored = fh.read().strip()
        if stored:
            return stored
    except OSError:
        pass

    key = secrets.token_hex(32)
    try:
        # O_EXCL: if another process created it a moment ago, fail here and
        # read theirs instead, so two workers never disagree on the key.
        fd = os.open(SECRET_KEY_FILE,
                     os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(key)
    except FileExistsError:
        try:
            with open(SECRET_KEY_FILE, encoding="utf-8") as fh:
                key = fh.read().strip() or key
        except OSError:
            pass
    except OSError:
        pass  # unwritable location: usable for this process, not persisted
    return key


def debug_enabled() -> bool:
    """True only when FLASK_DEBUG explicitly asks for the Werkzeug debugger.

    Off by default: while the debugger is active it exposes a console that
    can execute code on this machine. Set ``FLASK_DEBUG=1`` for development.
    """
    return os.environ.get("FLASK_DEBUG", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


class Config:
    SECRET_KEY = resolve_secret_key()
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp", "gif"}
    DATABASE_PATH = os.path.join(BASE_DIR, "app", "models", "designscore.db")

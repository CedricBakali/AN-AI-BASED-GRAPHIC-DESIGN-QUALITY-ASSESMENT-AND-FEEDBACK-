"""Security regressions.

These lock in the specific weaknesses that were found and fixed:
a published default session key, an interactive debugger left on, and a
disabled account that could keep browsing on an old session.
"""
from __future__ import annotations

import io
import pathlib

import pytest

import config
from config import debug_enabled

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _user_id(client) -> int:
    from app.models import database
    with database.get_connection() as conn:
        row = conn.execute(
            "SELECT id FROM users WHERE email = ?", ("tester@test.com",)
        ).fetchone()
    assert row is not None
    return int(row["id"])


# ------------------------------------------------------------- the key

def test_no_published_default_session_key_remains():
    """A public constant here would let anyone forge a session cookie."""
    for rel in ("app.py", "config.py"):
        src = (ROOT / rel).read_text(encoding="utf-8")
        assert "dev-secret-change-me" not in src, rel


def test_environment_variable_always_wins(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SECRET_KEY_FILE", str(tmp_path / "key"))
    monkeypatch.setenv("DESIGNSCORE_SECRET", "from-the-environment")
    assert config.resolve_secret_key() == "from-the-environment"
    # An explicit key is never written to disk.
    assert not (tmp_path / "key").exists()


def test_key_is_generated_once_then_reused(tmp_path, monkeypatch):
    path = tmp_path / "key"
    monkeypatch.setattr(config, "SECRET_KEY_FILE", str(path))
    monkeypatch.delenv("DESIGNSCORE_SECRET", raising=False)

    first = config.resolve_secret_key()
    assert len(first) >= 32, "a generated key must not be guessable"
    assert first != "dev-secret-change-me"

    second = config.resolve_secret_key()
    assert second == first, "sessions must survive a restart"
    assert path.read_text(encoding="utf-8").strip() == first


def test_app_uses_the_resolver_not_a_literal():
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "resolve_secret_key" in src
    assert "app.secret_key = resolve_secret_key()" in src


def test_the_key_file_is_gitignored():
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert ".designscore_secret" in ignore


# ------------------------------------------------------------- the debugger

def test_debugger_is_off_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("FLASK_DEBUG", raising=False)
    assert debug_enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_debugger_can_be_opted_into(value, monkeypatch):
    monkeypatch.setenv("FLASK_DEBUG", value)
    assert debug_enabled() is True


@pytest.mark.parametrize("value", ["0", "false", "no", "", " "])
def test_everything_else_keeps_it_off(value, monkeypatch):
    monkeypatch.setenv("FLASK_DEBUG", value)
    assert debug_enabled() is False


def test_app_never_hard_codes_debug_on():
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "debug=True" not in src


# ------------------------------------------------- disabled accounts

def test_disabled_account_is_logged_out_of_every_page(client, logged_client):
    """The session cookie alone must not outlive the account itself."""
    from app.models import database

    uid = _user_id(logged_client)
    # Prove the account works while it is active.
    assert logged_client.get("/history").status_code == 200

    with database.get_connection() as conn:
        conn.execute("UPDATE users SET is_active = 0 WHERE id = ?", (uid,))

    rv = logged_client.get("/history")
    assert rv.status_code == 302
    assert "/login" in rv.location


def test_disabled_account_cannot_perform_actions(client, logged_client):
    """Reading is blocked; writing must be blocked too.

    The update must be committed BEFORE the request runs - SQLite holds the
    write inside the connection's transaction until it closes, and a
    separate request connection would otherwise still see the account as
    active.
    """
    from app.models import database

    uid = _user_id(logged_client)
    with database.get_connection() as conn:
        before = conn.execute(
            "SELECT COUNT(*) AS n FROM analyses WHERE user_id = ?", (uid,)
        ).fetchone()["n"]

    with database.get_connection() as conn:
        conn.execute("UPDATE users SET is_active = 0 WHERE id = ?", (uid,))

    buf = io.BytesIO()
    from PIL import Image
    Image.new("RGB", (200, 160), (120, 30, 90)).save(buf, format="PNG")
    buf.seek(0)
    rv = logged_client.post(
        "/analyze", data={"image": (buf, "disabled.png")},
        content_type="multipart/form-data",
    )

    with database.get_connection() as conn:
        after = conn.execute(
            "SELECT COUNT(*) AS n FROM analyses WHERE user_id = ?", (uid,)
        ).fetchone()["n"]

    assert rv.status_code == 302
    assert "/login" in rv.headers.get("Location", "")
    assert after == before, "a disabled account must not create analyses"


def test_disabled_account_loses_the_admin_panel(client, logged_client):
    """admin_required already refused them; login_required now agrees."""
    from app.models import database

    uid = _user_id(logged_client)
    with database.get_connection() as conn:
        conn.execute("UPDATE users SET is_active = 0 WHERE id = ?", (uid,))

    rv = logged_client.get("/admin/")
    assert rv.status_code == 302
    assert "/login" in rv.location

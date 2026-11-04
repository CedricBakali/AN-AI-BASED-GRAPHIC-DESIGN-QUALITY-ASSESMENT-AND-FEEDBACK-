"""CSRF protection: every state-changing request must carry this
session's own token.

The harness normally lifts enforcement (conftest and the system test
both drive the app with test_client() and never render a form), so the
tests below switch it back on explicitly and prove three things:

  1. a form rendered by the app contains a usable token,
  2. a POST without it - or with somebody else's - is refused,
  3. a POST carrying the right token still works.
"""
from __future__ import annotations

import re

TOKEN_RE = re.compile(r'name="csrf_token"\s+value="([^"]+)"')


def _token(html: str) -> str:
    """Pull the hidden field out of a rendered page."""
    match = TOKEN_RE.search(html)
    assert match, "no csrf_token hidden field was rendered"
    return match.group(1)


def _enforce(app) -> None:
    """Undo the harness lift so the app behaves like production."""
    app.config["TESTING"] = False
    app.config["CSRF_ENABLED"] = True


def _still_anonymous(client) -> bool:
    """True when the client was NOT let in by the last POST."""
    rv = client.get("/dashboard")
    return rv.status_code == 302 and "/login" in rv.headers.get("Location", "")


# ------------------------------------------------------------------
# 1. The app hands out a token with every form.
# ------------------------------------------------------------------
def test_login_form_carries_a_token(client):
    assert len(_token(client.get("/login").data.decode())) >= 32


def test_register_form_carries_a_token(client):
    assert len(_token(client.get("/register").data.decode())) >= 32


def test_dashboard_form_carries_a_token(logged_client):
    assert len(_token(logged_client.get("/dashboard").data.decode())) >= 32


def test_profile_forms_carry_a_token(logged_client):
    html = logged_client.get("/profile").data.decode()
    # The password form on this page is the one state-changing POST.
    assert len(TOKEN_RE.findall(html)) >= 1


def test_label_forms_carry_a_token(client):
    """The admin labelling page posts three separate forms."""
    from app.models import database

    client.post("/register", data={
        "username": "root", "email": "root@test.com",
        "password": "password123",
    })
    with database.get_connection() as conn:
        conn.execute("UPDATE users SET role = 'admin' WHERE email = ?",
                     ("root@test.com",))
    client.post("/login", data={
        "email": "root@test.com", "password": "password123",
    })
    html = client.get("/admin/label/", follow_redirects=True).data.decode()
    assert len(TOKEN_RE.findall(html)) >= 3


# ------------------------------------------------------------------
# 2. The token belongs to exactly one session.
# ------------------------------------------------------------------
def test_token_is_stable_within_a_session(client):
    first = _token(client.get("/login").data.decode())
    second = _token(client.get("/login").data.decode())
    assert first == second


def test_token_differs_between_sessions(client):
    other = client.application.test_client()
    assert _token(client.get("/login").data.decode()) != \
        _token(other.get("/login").data.decode())


# ------------------------------------------------------------------
# 3. Enforcement: refused without a token, accepted with one.
# ------------------------------------------------------------------
def _register(client) -> None:
    """Create the account while the harness lift is still on, so the
    login attempt below is refused for CSRF reasons and not because
    the account happens to be missing."""
    client.post("/register", data={
        "username": "tester", "email": "tester@test.com",
        "password": "password123",
    })


def test_login_without_a_token_is_refused(client, test_app):
    _register(client)
    _enforce(test_app)
    client.post("/login", data={
        "email": "tester@test.com", "password": "password123",
    })
    assert _still_anonymous(client)


def test_login_with_a_forged_token_is_refused(client, test_app):
    _register(client)
    _enforce(test_app)
    client.post("/login", data={
        "email": "tester@test.com", "password": "password123",
        "csrf_token": "not-the-right-token",
    })
    assert _still_anonymous(client)


def test_login_with_the_session_token_is_accepted(client, test_app):
    _register(client)
    _enforce(test_app)
    token = _token(client.get("/login").data.decode())
    rv = client.post("/login", data={
        "email": "tester@test.com", "password": "password123",
        "csrf_token": token,
    })
    assert rv.status_code == 302
    assert not _still_anonymous(client)


def test_somebody_elses_token_is_refused(client, test_app):
    """A token lifted from another browser must not work here."""
    _register(client)
    _enforce(test_app)

    intruder = client.application.test_client()
    stolen = _token(intruder.get("/login").data.decode())
    own = _token(client.get("/login").data.decode())
    assert own != stolen

    rv = client.post("/login", data={
        "email": "tester@test.com", "password": "password123",
        "csrf_token": stolen,
    })
    assert rv.status_code == 302
    assert _still_anonymous(client)


def test_logged_in_post_without_a_token_is_refused(logged_client, test_app):
    """A live session does not excuse a token-less state change."""
    _enforce(test_app)
    rv = logged_client.post("/change_password", data={
        "current_password": "password123", "new_password": "password456",
    })
    assert rv.status_code == 302
    # The profile page explains what happened instead of applying it.
    page = logged_client.get("/profile").data.decode()
    assert "session expired" in page
    assert "password456" not in page  # never applied


def test_ajax_post_without_a_token_gets_json_400(client, test_app):
    """The uploader asks for JSON, so it gets JSON - not a redirect."""
    _enforce(test_app)
    rv = client.post("/login", data={}, headers={
        "X-Requested-With": "XMLHttpRequest",
    })
    assert rv.status_code == 400
    payload = rv.get_json()
    assert payload["ok"] is False
    assert payload["message"]


# ------------------------------------------------------------------
# 4. Safe methods are never challenged.
# ------------------------------------------------------------------
def test_get_requests_are_never_challenged(client, test_app):
    _enforce(test_app)
    assert client.get("/login").status_code == 200
    assert client.get("/register").status_code == 200
    assert client.get("/").status_code in (200, 302)


def test_get_logout_still_works_under_enforcement(client, test_app):
    """GET /logout is a known, accepted leftover - it must not break."""
    _enforce(test_app)
    assert client.get("/logout").status_code == 302

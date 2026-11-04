"""Uploaded designs must not be readable by strangers.

Before this, Flask's default static handler served everything in
``static/uploads/`` with no authentication at all: knowing (or guessing)
a file name was enough to read somebody else's design. These tests pin
down the ownership check that now runs in front of that folder.

Every access test runs through ``uploads_served`` so that a 404 means
"refused" rather than "file not found" - the fixtures write uploads into
a temp folder, while Flask's static handler looks in the real one.
"""
from __future__ import annotations

import io

import pytest

from app.models import database


# ------------------------------------------------------------- fixtures

@pytest.fixture()
def uploads_served(test_app, tmp_path, monkeypatch):
    """Make ``/static/<path>`` resolve inside the isolated uploads folder.

    ``conftest`` points ``UPLOAD_DIR`` at ``tmp_path/uploads`` so pytest
    never writes into the repository, but Flask still serves ``/static``
    from the project folder. Pointing ``static_folder`` at ``tmp_path``
    makes ``/static/uploads/<name>`` land on the file that was actually
    uploaded - so the status code then reflects authorisation, not a
    missing file.
    """
    monkeypatch.setattr(test_app, "static_folder", str(tmp_path))
    return test_app


def _signup(client, email: str, username: str):
    client.post("/register", data={"username": username, "email": email,
                                   "password": "password123"})
    client.post("/login", data={"email": email, "password": "password123"})
    return client


@pytest.fixture()
def other_client(test_app):
    """A second session belonging to a different, unrelated person."""
    return _signup(test_app.test_client(), "other@test.com", "other")


@pytest.fixture()
def admin_client(test_app):
    """A separate session whose account has been promoted to administrator."""
    client = _signup(test_app.test_client(), "boss@test.com", "boss")
    with database.get_connection() as conn:
        conn.execute("UPDATE users SET role = 'admin' WHERE email = ?",
                     ("boss@test.com",))
    return client


def _owned_url(client) -> str:
    """Upload a design as ``client`` and return the URL it is served from."""
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (240, 180), (30, 120, 200)).save(buf, format="PNG")
    buf.seek(0)

    rv = client.post("/analyze", data={"image": (buf, "private.png")},
                     content_type="multipart/form-data",
                     follow_redirects=False)
    assert rv.status_code == 302, rv.status_code

    analysis_id = int(rv.location.rsplit("/", 1)[1])
    with database.get_connection() as conn:
        row = conn.execute(
            "SELECT image_path FROM analyses WHERE id = ?", (analysis_id,)
        ).fetchone()
    assert row is not None
    assert row["image_path"].startswith("static/uploads/")
    return "/" + row["image_path"]


# ------------------------------------------------------------- the rules

def test_ordinary_static_assets_are_untouched(client):
    """CSS and JS must stay public - only uploads are guarded."""
    assert client.get("/static/css/report.css").status_code == 200


def test_owner_can_read_their_own_design(logged_client, uploads_served):
    url = _owned_url(logged_client)
    rv = logged_client.get(url)
    assert rv.status_code == 200, "the file exists, so this is a refusal"
    assert rv.mimetype.startswith("image/")


def test_an_existing_file_is_hidden_from_strangers(logged_client,
                                                   uploads_served):
    """The decisive case: the file is provably reachable, then provably not."""
    url = _owned_url(logged_client)

    # Reachable by its owner...
    assert logged_client.get(url).status_code == 200

    # ...and invisible to a stranger with a valid session.
    stranger = uploads_served.test_client()
    stranger.post("/register", data={"username": "sneak",
                                     "email": "sneak@test.com",
                                     "password": "password123"})
    stranger.post("/login", data={"email": "sneak@test.com",
                                  "password": "password123"})
    assert stranger.get(url).status_code == 404


def test_anonymous_visitor_reads_nothing(logged_client, uploads_served):
    url = _owned_url(logged_client)
    anonymous = uploads_served.test_client()
    assert anonymous.get(url).status_code == 404


def test_another_user_reads_nothing(logged_client, other_client,
                                    uploads_served):
    url = _owned_url(logged_client)
    assert other_client.get(url).status_code == 404


def test_each_person_only_sees_their_own(logged_client, other_client,
                                         uploads_served):
    mine = _owned_url(logged_client)
    theirs = _owned_url(other_client)

    assert logged_client.get(mine).status_code == 200
    assert other_client.get(theirs).status_code == 200
    assert logged_client.get(theirs).status_code == 404
    assert other_client.get(mine).status_code == 404


def test_admin_can_review_any_design(logged_client, admin_client,
                                     uploads_served):
    """The labelling page has to be able to show every submitted design."""
    url = _owned_url(logged_client)
    assert admin_client.get(url).status_code == 200


def test_a_disabled_account_loses_their_images_immediately(logged_client,
                                                          uploads_served):
    url = _owned_url(logged_client)
    assert logged_client.get(url).status_code == 200

    with database.get_connection() as conn:
        row = conn.execute(
            "SELECT id FROM users WHERE email = ?", ("tester@test.com",)
        ).fetchone()
        conn.execute("UPDATE users SET is_active = 0 WHERE id = ?",
                     (row["id"],))

    assert logged_client.get(url).status_code == 404


def test_an_orphan_upload_belongs_to_nobody(logged_client, uploads_served):
    """A file on disk with no analysis row is not readable by a user."""
    url = _owned_url(logged_client)
    assert logged_client.get(url).status_code == 200

    with database.get_connection() as conn:
        conn.execute("DELETE FROM analyses WHERE image_path = ?",
                     ("static/uploads/" + url.rsplit("/", 1)[-1],))

    assert logged_client.get(url).status_code == 404


@pytest.mark.parametrize("path", [
    "/static/uploads/",
    "/static/uploads/sub/level.png",
    "/static/uploads/../models/designscore.db",
    "/static/uploads/..%2f..%2fapp.py",
    "/static/uploads/%2e%2e%2fapp.py",
    "/static/uploads/%2e%2e/%2e%2e/app.py",
])
def test_traversal_and_nesting_are_refused(client, uploads_served, path):
    """Nothing may be reached that is not a flat name inside the folder."""
    assert client.get(path).status_code == 404


def test_a_refusal_is_a_404_not_a_403(logged_client, other_client,
                                      uploads_served):
    """Never confirm that somebody else's file exists."""
    url = _owned_url(logged_client)
    rv = other_client.get(url)
    assert rv.status_code == 404
    assert rv.status_code != 403

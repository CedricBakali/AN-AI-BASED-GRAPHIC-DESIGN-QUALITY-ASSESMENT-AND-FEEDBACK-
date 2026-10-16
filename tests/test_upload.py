"""Tests for upload validation + analysis flow (Modules 2, 6-8)."""
from __future__ import annotations

import io

from PIL import Image

from app.routes.analysis import allowed_file


def test_allowed_file_helper():
    assert allowed_file("poster.png")
    assert allowed_file("flyer.JPG")
    assert not allowed_file("notes.txt")
    assert not allowed_file("virus.exe")
    assert not allowed_file("noextension")


def _image_bytes(color=(30, 120, 200), size=(400, 300)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    buf.seek(0)
    return buf


def test_upload_analyze_result_history(logged_client):
    rv = logged_client.post("/analyze", data={
        "image": (_image_bytes(), "poster.png"),
    }, content_type="multipart/form-data", follow_redirects=False)
    assert rv.status_code == 302
    assert rv.location.startswith("/result/")
    page = logged_client.get(rv.location)
    assert page.status_code == 200
    html = page.data.decode()
    for needle in ("Strengths", "Weaknesses", "Recommendations",
                   "Category scores", "Layout", "Visual Hierarchy"):
        assert needle in html
    history = logged_client.get("/history")
    assert history.status_code == 200
    assert "poster.png" in history.data.decode()


def test_upload_rejects_bad_type(logged_client):
    rv = logged_client.post("/analyze", data={
        "image": (io.BytesIO(b"not an image"), "notes.txt"),
    }, content_type="multipart/form-data", follow_redirects=False)
    assert rv.status_code == 302
    assert rv.location.endswith("/dashboard")


def test_upload_requires_file(logged_client):
    rv = logged_client.post("/analyze", data={},
                            follow_redirects=False)
    assert rv.status_code == 302


def test_users_cannot_see_each_others_results(client, logged_client):
    rv = logged_client.post("/analyze", data={
        "image": (_image_bytes(), "mine.png"),
    }, content_type="multipart/form-data", follow_redirects=False)
    other = client
    other.post("/register", data={
        "username": "other", "email": "other@test.com",
        "password": "password123",
    })
    other.post("/login", data={
        "email": "other@test.com", "password": "password123",
    })
    assert other.get(rv.location).status_code == 404
    assert "mine.png" not in other.get("/history").data.decode()

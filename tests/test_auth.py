"""Tests for registration, login, guards (Module 1)."""
from __future__ import annotations


def test_register_and_login(client):
    assert client.post("/register", data={
        "username": "u1", "email": "u1@test.com", "password": "password123",
    }).status_code == 302
    rv = client.post("/login", data={
        "email": "u1@test.com", "password": "password123",
    }, follow_redirects=False)
    assert rv.status_code == 302
    assert rv.location.endswith("/dashboard")
    assert client.get("/dashboard").status_code == 200


def test_register_defaults_to_normal_user(client):
    from app.models.user import get_user_by_email
    client.post("/register", data={
        "username": "u2", "email": "u2@test.com", "password": "password123",
    })
    row = get_user_by_email("u2@test.com")
    assert row["role"] == "user"
    assert row["is_active"] == 1


def test_register_rejects_duplicate_and_short_password(client):
    data = {"username": "u3", "email": "u3@test.com", "password": "password123"}
    client.post("/register", data=data)
    dup = client.post("/register", data=data, follow_redirects=False)
    assert dup.status_code == 302  # back to register with flash
    short = client.post("/register", data={
        "username": "u4", "email": "u4@test.com", "password": "short",
    }, follow_redirects=False)
    assert short.status_code == 302
    bad = client.post("/login", data={
        "email": "u3@test.com", "password": "wrongpass1",
    }, follow_redirects=False)
    assert bad.status_code == 302
    assert bad.location.endswith("/login")


def test_protected_routes_need_login(client):
    assert client.get("/dashboard").status_code == 302
    assert client.get("/history").status_code == 302
    assert client.post("/analyze", data={}).status_code == 302


def test_logout_clears_session(logged_client):
    assert logged_client.get("/dashboard").status_code == 200
    logged_client.get("/logout")
    assert logged_client.get("/dashboard").status_code == 302

"""Shared pytest fixtures: isolated DB + Flask test client.

Each test gets a FRESH temporary database so the real
app/models/designscore.db is never touched.
"""
from __future__ import annotations

import importlib.util
import pathlib

import pytest

from app.models import database

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _load_app():
    spec = importlib.util.spec_from_file_location(
        "main_app_under_test", str(ROOT / "app.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.app.config.update(TESTING=True)
    return module.app


@pytest.fixture()
def tmp_db(tmp_path, monkeypatch):
    """Point the database module at a temp file and init schema."""
    monkeypatch.setattr(database, "DB_PATH", str(tmp_path / "test.db"))
    database.init_db()
    return database


@pytest.fixture()
def test_app(tmp_db):
    return _load_app()


@pytest.fixture()
def client(test_app):
    return test_app.test_client()


@pytest.fixture()
def logged_client(client):
    client.post("/register", data={
        "username": "tester",
        "email": "tester@test.com",
        "password": "password123",
    })
    client.post("/login", data={
        "email": "tester@test.com",
        "password": "password123",
    })
    return client

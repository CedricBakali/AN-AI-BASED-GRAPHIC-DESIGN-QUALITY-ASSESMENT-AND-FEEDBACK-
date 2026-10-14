from __future__ import annotations

from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import sqlite3

from app.models.database import get_connection


# =========================================================
# PASSWORD FUNCTIONS
# =========================================================

def hash_password(password: str) -> str:
    """
    Convert a plain-text password into a secure password hash.
    """

    return generate_password_hash(password)


def check_password(password: str, password_hash: str) -> bool:
    """
    Check whether the entered password matches
    the stored password hash.
    """

    return check_password_hash(password_hash, password)


# =========================================================
# CREATE USER
# =========================================================

def create_user(
    username: str,
    email: str,
    password: str
) -> int:
    """
    Create a normal user account.

    New accounts automatically receive:
        role = 'user'
        is_active = 1

    Returns:
        int: ID of the newly created user.
    """

    password_hash = hash_password(password)

    with get_connection() as conn:

        cursor = conn.execute(
            """
            INSERT INTO users (
                username,
                email,
                password_hash,
                role,
                is_active,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                username,
                email,
                password_hash,
                "user",
                1,
                datetime.utcnow().isoformat(timespec="seconds")
            )
        )

        return int(cursor.lastrowid)


# =========================================================
# GET USER BY EMAIL
# =========================================================

def get_user_by_email(email: str):
    """
    Find a user using their email address.

    Returns:
        sqlite3.Row or None
    """

    with get_connection() as conn:

        return conn.execute(
            """
            SELECT *
            FROM users
            WHERE email = ?
            """,
            (email,)
        ).fetchone()


# =========================================================
# GET USER BY ID
# =========================================================

def get_user_by_id(user_id: int):
    """
    Find a user using their database ID.

    Returns:
        sqlite3.Row or None
    """

    with get_connection() as conn:

        return conn.execute(
            """
            SELECT *
            FROM users
            WHERE id = ?
            """,
            (user_id,)
        ).fetchone()


# =========================================================
# CHECK IF EMAIL EXISTS
# =========================================================

def email_exists(email: str) -> bool:
    """
    Check whether an email address is already registered.
    """

    with get_connection() as conn:

        row = conn.execute(
            """
            SELECT id
            FROM users
            WHERE email = ?
            """,
            (email,)
        ).fetchone()

        return row is not None


# =========================================================
# CHECK IF USER IS ACTIVE
# =========================================================

def is_user_active(user_id: int) -> bool:
    """
    Check whether a user's account is active.
    """

    with get_connection() as conn:

        row = conn.execute(
            """
            SELECT is_active
            FROM users
            WHERE id = ?
            """,
            (user_id,)
        ).fetchone()

        if row is None:
            return False

        return bool(row["is_active"])


# =========================================================
# GET USER ROLE
# =========================================================

def get_user_role(user_id: int) -> str | None:
    """
    Return the role of a user.

    Possible values:
        user
        admin
    """

    with get_connection() as conn:

        row = conn.execute(
            """
            SELECT role
            FROM users
            WHERE id = ?
            """,
            (user_id,)
        ).fetchone()

        if row is None:
            return None

        return row["role"]
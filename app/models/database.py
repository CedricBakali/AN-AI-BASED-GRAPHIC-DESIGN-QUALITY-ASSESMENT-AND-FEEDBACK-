"""SQLite persistence for DesignScore."""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime
from contextlib import contextmanager


# =========================================================
# DATABASE LOCATION
# =========================================================

DB_PATH = os.path.join(
    os.path.dirname(__file__),
    "designscore.db"
)


# =========================================================
# DATABASE CONNECTION
# =========================================================

@contextmanager
def get_connection():
    conn = sqlite3.connect(DB_PATH)

    # Allows us to access columns by name
    conn.row_factory = sqlite3.Row

    # Enable foreign key relationships
    conn.execute("PRAGMA foreign_keys = ON")

    try:
        yield conn
        conn.commit()

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()


# =========================================================
# CREATE DATABASE TABLES
# =========================================================

def _ensure_column(conn, table: str, column: str, definition: str) -> None:
    cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def init_db() -> None:

    with get_connection() as conn:

        # -----------------------------
        # USERS TABLE
        # -----------------------------

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                username      TEXT    NOT NULL,
                email         TEXT    NOT NULL UNIQUE,
                password_hash TEXT    NOT NULL,
                role          TEXT    NOT NULL DEFAULT 'user'
                              CHECK (role IN ('user', 'admin')),
                is_active     INTEGER NOT NULL DEFAULT 1,
                created_at    TEXT    NOT NULL
            )
            """
        )

        # Migrate stale DBs created before role/is_active existed.
        _ensure_column(conn, "users", "role", "TEXT NOT NULL DEFAULT 'user'")
        _ensure_column(conn, "users", "is_active", "INTEGER NOT NULL DEFAULT 1")

        # -----------------------------
        # ANALYSES TABLE
        # -----------------------------

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS analyses (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,

                user_id     INTEGER NOT NULL
                            REFERENCES users(id)
                            ON DELETE CASCADE,

                filename    TEXT    NOT NULL,
                image_path  TEXT    NOT NULL,

                overall     INTEGER NOT NULL,
                grade       TEXT    NOT NULL,

                palette     TEXT    NOT NULL,
                categories  TEXT    NOT NULL,

                created_at  TEXT    NOT NULL
            )
            """
        )

        # Old DBs lack user_id (nullable so existing rows survive;
        # new rows always supply it, per-user queries ignore orphans).
        _ensure_column(conn, "analyses", "user_id", "INTEGER REFERENCES users(id) ON DELETE CASCADE")


# =========================================================
# SAVE ANALYSIS
# =========================================================

def save_analysis(
    user_id: int,
    filename: str,
    image_path: str,
    result
) -> int:
    """
    Save an analysis belonging to a specific user.

    Returns:
        int: ID of the newly created analysis.
    """

    data = result.as_dict()

    with get_connection() as conn:

        cursor = conn.execute(
            """
            INSERT INTO analyses (
                user_id,
                filename,
                image_path,
                overall,
                grade,
                palette,
                categories,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                filename,
                image_path,
                data["overall"],
                data["grade"],
                json.dumps(data["palette"]),
                json.dumps(data["categories"]),
                datetime.utcnow().isoformat(timespec="seconds")
            )
        )

        return int(cursor.lastrowid)


# =========================================================
# CONVERT DATABASE ROW TO DICTIONARY
# =========================================================

def _row_to_dict(row: sqlite3.Row) -> dict:

    return {
        "id": row["id"],
        "user_id": row["user_id"],
        "filename": row["filename"],
        "image_path": row["image_path"],
        "overall": row["overall"],
        "grade": row["grade"],
        "palette": json.loads(row["palette"]),
        "categories": json.loads(row["categories"]),
        "created_at": row["created_at"]
    }


# =========================================================
# GET USER'S ASSESSMENT HISTORY
# =========================================================

def get_history(
    user_id: int,
    limit: int = 20
) -> list[dict]:
    """
    Get the most recent assessments belonging
    to the logged-in user.
    """

    with get_connection() as conn:

        rows = conn.execute(
            """
            SELECT *
            FROM analyses
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (user_id, limit)
        ).fetchall()

        return [
            _row_to_dict(row)
            for row in rows
        ]


# =========================================================
# GET ONE ANALYSIS
# =========================================================
def get_analysis(
    analysis_id: int,
    user_id: int
) -> dict | None:
    """
    Get one assessment belonging to a specific user.

    The user_id check prevents one user from
    viewing another user's assessment.
    """

    with get_connection() as conn:

        row = conn.execute(
            """
            SELECT *
            FROM analyses
            WHERE id = ?
              AND user_id = ?
            """,
            (analysis_id, user_id)
        ).fetchone()

        if row is None:
            return None

        return _row_to_dict(row)


# =========================================================
# ADMIN: SYSTEM STATISTICS
# =========================================================

def get_stats() -> dict:
    """Counts and averages across ALL users (admin only)."""

    with get_connection() as conn:

        total_users = conn.execute(
            "SELECT COUNT(*) FROM users"
        ).fetchone()[0]

        active_users = conn.execute(
            "SELECT COUNT(*) FROM users WHERE is_active = 1"
        ).fetchone()[0]

        total_assessments = conn.execute(
            "SELECT COUNT(*) FROM analyses"
        ).fetchone()[0]

        avg_overall = conn.execute(
            "SELECT AVG(overall) FROM analyses"
        ).fetchone()[0]

        return {
            "total_users": total_users,
            "active_users": active_users,
            "total_assessments": total_assessments,
            "average_score": round(avg_overall) if avg_overall is not None else None,
        }


# =========================================================
# ADMIN: ALL USERS WITH ASSESSMENT COUNTS
# =========================================================

def get_all_users_with_counts() -> list[dict]:
    """Every user plus how many assessments each owns (admin only)."""

    with get_connection() as conn:

        rows = conn.execute(
            """
            SELECT u.id, u.username, u.email, u.role,
                   u.is_active, u.created_at,
                   COUNT(a.id) AS assessment_count
            FROM users u
            LEFT JOIN analyses a ON a.user_id = u.id
            GROUP BY u.id
            ORDER BY u.id
            """
        ).fetchall()

        return [dict(row) for row in rows]


# =========================================================
# ADMIN: RECENT ASSESSMENTS ACROSS ALL USERS
# =========================================================

def get_recent_assessments(limit: int = 10) -> list[dict]:
    """Latest assessments from every user (admin only)."""

    with get_connection() as conn:

        rows = conn.execute(
            """
            SELECT *
            FROM analyses
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,)
        ).fetchall()

        return [
            _row_to_dict(row)
            for row in rows
        ]
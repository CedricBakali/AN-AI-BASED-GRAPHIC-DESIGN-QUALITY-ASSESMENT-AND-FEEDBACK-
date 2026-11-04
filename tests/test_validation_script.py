"""scripts/validate_clip.py must fail honestly instead of inventing data.

The project deliberately ships no dataset and no human ratings, so the
one unacceptable behaviour for this script is printing a plausible
correlation out of thin air. These tests pin that down.
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "validate_clip.py"


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=str(ROOT), capture_output=True, text=True, timeout=300,
    )


def _seed(db_path: pathlib.Path, count: int) -> list[int]:
    """Insert `count` analyses that each carry a stored CLIP score.

    DB_PATH is restored on the way out so this subprocess-oriented test
    cannot leak its temporary path into the rest of the suite.
    """
    from app.models import database

    original = database.DB_PATH
    database.DB_PATH = str(db_path)
    ids: list[int] = []
    try:
        database.init_db()
        with database.get_connection() as conn:
            cursor = conn.execute(
                "INSERT INTO users (username, email, password_hash, role, "
                "is_active, created_at) VALUES (?, ?, ?, 'user', 1, ?)",
                ("v", "v@test.com", "not-a-real-hash", "2026-01-01")
            )
            user_id = cursor.lastrowid
            for index in range(count):
                cursor = conn.execute(
                    "INSERT INTO analyses (user_id, filename, image_path, "
                    "overall, grade, palette, categories, ai_score, "
                    "created_at) VALUES (?, ?, ?, 70, 'C', '[]', '{}', ?, ?)",
                    (user_id, f"design{index}.png",
                     f"/tmp/design{index}.png",
                     20 + index * 10, "2026-01-01")
                )
                ids.append(int(cursor.lastrowid))
    finally:
        database.DB_PATH = original
    return ids


# ------------------------------------------------------------------
# Refusals - the whole point of the scaffold.
# ------------------------------------------------------------------
def test_without_a_ratings_file_it_refuses(tmp_path):
    result = _run("--ratings", str(tmp_path / "ratings.csv"))
    assert result.returncode == 1
    assert "CANNOT VALIDATE" in result.stderr
    assert "no dataset" in result.stderr
    assert "Spearman" not in result.stdout  # no number without data


def test_it_refuses_rather_than_writes_a_ratings_file(tmp_path):
    target = tmp_path / "ratings.csv"
    _run("--ratings", str(target))
    assert not target.exists(), "the script must never author ratings"


def test_a_csv_without_a_key_column_is_refused(tmp_path):
    ratings = tmp_path / "ratings.csv"
    ratings.write_text("image_id,comments\n1,looks nice\n",
                       encoding="utf-8")
    result = _run("--ratings", str(ratings))
    assert result.returncode == 1
    assert "no key column" in result.stderr
    assert "Spearman" not in result.stdout


def test_a_csv_without_a_rating_column_is_refused(tmp_path):
    ratings = tmp_path / "ratings.csv"
    ratings.write_text("analysis_id,notes\n42,fine\n", encoding="utf-8")
    result = _run("--ratings", str(ratings))
    assert result.returncode == 1
    assert "no rating column" in result.stderr
    assert "Spearman" not in result.stdout


def test_empty_ratings_are_refused(tmp_path):
    ratings = tmp_path / "ratings.csv"
    ratings.write_text("analysis_id,rating\n", encoding="utf-8")
    result = _run("--ratings", str(ratings))
    assert result.returncode == 1
    assert "no usable rows" in result.stderr
    assert "Spearman" not in result.stdout


def test_too_few_paired_samples_is_refused(tmp_path):
    db = tmp_path / "seed.db"
    ids = _seed(db, 2)
    ratings = tmp_path / "ratings.csv"
    ratings.write_text(
        "analysis_id,rating\n" + "".join(f"{i},50\n" for i in ids),
        encoding="utf-8")

    result = _run("--ratings", str(ratings), "--db", str(db))
    assert result.returncode == 1
    assert "at least 5 are needed" in result.stderr
    assert "Spearman" not in result.stdout


def test_unmatched_images_are_refused(tmp_path):
    db = tmp_path / "seed.db"
    _seed(db, 8)
    ratings = tmp_path / "ratings.csv"
    # Ratings exist, but for designs that were never analysed here.
    ratings.write_text("analysis_id,rating\n"
                       + "".join(f"{900 + i},50\n" for i in range(8)),
                       encoding="utf-8")

    result = _run("--ratings", str(ratings), "--db", str(db))
    assert result.returncode == 1
    assert "none of the rated images" in result.stderr
    assert "Spearman" not in result.stdout


def test_missing_database_is_refused(tmp_path):
    ratings = tmp_path / "ratings.csv"
    ratings.write_text("analysis_id,rating\n1,50\n", encoding="utf-8")
    result = _run("--ratings", str(ratings),
                  "--db", str(tmp_path / "nope.db"))
    assert result.returncode == 1
    assert "no database" in result.stderr


# ------------------------------------------------------------------
# The happy path exists - but no expected correlation value is pinned,
# because any number this test could assert would be synthetic.
# ------------------------------------------------------------------
def test_pairs_and_reports_when_data_is_real(tmp_path):
    db = tmp_path / "seed.db"
    ids = _seed(db, 6)
    ratings = tmp_path / "ratings.csv"
    ratings.write_text(
        "analysis_id,rating\n"
        + "".join(f"{i},{40 + n * 5}\n" for n, i in enumerate(ids)),
        encoding="utf-8")

    result = _run("--ratings", str(ratings), "--db", str(db),
                  "--bootstrap", "200", "--seed", "1")
    assert result.returncode == 0, result.stderr
    assert "Paired samples   : 6" in result.stdout
    assert "Spearman rho" in result.stdout
    assert "Pearson r" in result.stdout
    assert "95% CI" in result.stdout
    # Explicitly separated from the rule-based score in the output.
    assert "does not validate the 0-100 Design Quality Score" in result.stdout

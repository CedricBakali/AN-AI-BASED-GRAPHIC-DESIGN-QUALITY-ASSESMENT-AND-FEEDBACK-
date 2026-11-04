"""VALIDATION SCAFFOLD - measure how well CLIP agrees with human ratings.

This script answers one question honestly:

    When people rate a design, does CLIP's AI Visual Assessment agree
    with them?

It does NOT invent anything. It reads human ratings that *you* supply in
``validation/ratings.csv``, pairs each one with the CLIP score already
stored for that analysis, and prints the correlation with a bootstrap
confidence interval.

.. warning::
   This repository ships **no dataset, no human ratings and no
   correlation results**. Running this script without a ratings file
   fails with an explanation rather than printing a plausible-looking
   number. Nothing here is generated, sampled or approximated to make
   the output look complete.

Usage:
    python scripts/validate_clip.py
    python scripts/validate_clip.py --ratings my_ratings.csv
    python scripts/validate_clip.py --bootstrap 20000 --seed 7

Expected CSV format (header required):

    analysis_id,rating
    42,85
    43,60

or, if you prefer to key on the uploaded filename:

    filename,rating
    poster_v2.png,85
    flyer_final.jpg,60

Optional columns:
    rater   - who rated it. Several rows with the same image are
              averaged first, which is ordinary aggregation of the
              ratings you supplied, not new data.
    rating  may also be spelled ``score``.

Exit codes:
    0  correlations printed
    1  cannot compute - no data, no matches, or too few pairs
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from app.models import database  # noqa: E402

DEFAULT_RATINGS = os.path.join("validation", "ratings.csv")

# Below this many paired observations a correlation is not worth
# reporting: the interval swallows everything.
MIN_PAIRS = 5


# =========================================================
# CORRELATION (numpy only - no scipy dependency)
# =========================================================
def _average_ranks(values: np.ndarray) -> np.ndarray:
    """1-based ranks, ties given their average rank (Spearman's rule)."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = average
        i = j + 1
    return ranks


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson r, or NaN when either side is constant."""
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.sqrt((x * x).sum() * (y * y).sum()))
    if denom == 0.0:
        return float("nan")
    return float((x * y).sum() / denom)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rho - Pearson correlation of the two rank columns."""
    return _pearson(_average_ranks(x), _average_ranks(y))


def bootstrap_ci(x: np.ndarray, y: np.ndarray, statistic,
                 draws: int, seed: int) -> tuple[float, float]:
    """Percentile 95% CI by resampling pairs with replacement."""
    rng = np.random.default_rng(seed)
    n = len(x)
    stats: list[float] = []
    for _ in range(draws):
        idx = rng.integers(0, n, n)
        value = statistic(x[idx], y[idx])
        if not np.isnan(value):
            stats.append(value)
    if len(stats) < 2:
        return float("nan"), float("nan")
    low, high = np.percentile(np.asarray(stats), [2.5, 97.5])
    return float(low), float(high)


# =========================================================
# INPUT
# =========================================================
def fail(message: str, *hints: str) -> None:
    """Refuse to produce numbers we cannot justify."""
    sys.stdout.flush()  # keep the banner above the explanation
    print(f"\nCANNOT VALIDATE: {message}", file=sys.stderr)
    for hint in hints:
        print(f"  - {hint}", file=sys.stderr)
    sys.exit(1)


def read_ratings(path: str) -> dict[str, list[float]]:
    """image key -> the human ratings supplied for it."""
    if not os.path.isfile(path):
        fail(
            f"no ratings file at {path!r}.",
            "This project deliberately ships no dataset and no human "
            "ratings, so there is nothing to validate against yet.",
            "Collect ratings yourself and save them as CSV, e.g. "
            "validation/ratings.csv with columns: analysis_id,rating",
            "See the module docstring of this script for the format.",
        )

    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            fail(f"{path} has no header row.",
                 "Expected a header such as: analysis_id,rating")
        fields = {name.strip().lower() for name in reader.fieldnames}

        key_column = None
        for candidate in ("analysis_id", "filename", "image", "image_path"):
            if candidate in fields:
                key_column = candidate
                break
        if key_column is None:
            fail(f"{path} has no key column.",
                 "Include one of: analysis_id, filename, image, image_path")

        if "rating" in fields:
            rating_column = "rating"
        elif "score" in fields:
            rating_column = "score"
        else:
            fail(f"{path} has no rating column.",
                 "Include a 'rating' column (0-100) with the human score.")

        ratings: dict[str, list[float]] = {}
        rows = 0
        for row in reader:
            raw_key = (row.get(key_column) or "").strip()
            raw_rating = (row.get(rating_column) or "").strip()
            if not raw_key or not raw_rating:
                continue
            try:
                value = float(raw_rating)
            except ValueError:
                fail(f"{path} row {rows + 2}: {raw_rating!r} is not a number.",
                     "Every rating must be numeric.")
            ratings.setdefault(raw_key, []).append(value)
            rows += 1

    if rows == 0:
        fail(f"{path} contained no usable rows.",
             "Expected data rows such as: 42,85")
    return ratings


def stored_ai_scores(db_path: str | None) -> tuple[dict[str, float], int]:
    """(image key -> CLIP score, number of analyses), keyed BOTH ways.

    A CSV may name the row by ``analysis_id`` or by ``filename``, so
    both spellings are accepted. A filename is only a fallback: if the
    same file was analysed twice, analysis_id is unambiguous.
    """
    if db_path:
        database.DB_PATH = db_path
    if not os.path.isfile(database.DB_PATH):
        fail(f"no database at {database.DB_PATH!r}.",
             "Run the app and analyse a few designs first.")

    scores: dict[str, float] = {}
    with database.get_connection() as conn:
        rows = conn.execute(
            "SELECT id, filename, ai_score FROM analyses "
            "WHERE ai_score IS NOT NULL"
        ).fetchall()
    for row in rows:
        value = float(row["ai_score"])
        scores[str(row["id"])] = value
        scores[str(row["filename"])] = value  # last duplicate wins
    if not rows:
        fail("no stored CLIP scores in the database.",
             "The AI Visual Assessment is NULL for every analysis - the "
             "model was unavailable when they were created.",
             "Re-run an analysis with CLIP available, then validate.")
    return scores, len(rows)


# =========================================================
# MAIN
# =========================================================
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Correlate CLIP's stored scores with human ratings "
                    "you supply. Prints nothing until real ratings exist.")
    parser.add_argument("--ratings", default=DEFAULT_RATINGS,
                        help="path to the human-ratings CSV "
                             f"(default: {DEFAULT_RATINGS})")
    parser.add_argument("--db", default=None,
                        help="path to designscore.db (default: the app's)")
    parser.add_argument("--bootstrap", type=int, default=10_000,
                        help="bootstrap resamples for the 95%% CI "
                             "(default: 10000)")
    parser.add_argument("--seed", type=int, default=0,
                        help="bootstrap seed, so a run is reproducible "
                             "(default: 0)")
    args = parser.parse_args()

    print("DesignScore - CLIP vs. human ratings")
    print("=" * 52)

    supplied = read_ratings(args.ratings)
    print(f"Ratings supplied : {args.ratings} ({sum(len(v) for v in supplied.values())} rows, "
          f"{len(supplied)} distinct images)")

    stored, stored_count = stored_ai_scores(args.db)
    print(f"Stored CLIP scores: {stored_count} analyses")

    paired_keys = [key for key in supplied if key in stored]
    if not paired_keys:
        available = sorted(key for key in stored if key.isdigit())
        fail("none of the rated images has a stored CLIP score.",
             "The CSV key must match an analysed design: use analysis_id "
             "(safest) or the exact filename.",
             "Analyses with a CLIP score: " + ", ".join(available))

    # Average repeated raters per image - aggregation of supplied data.
    human = np.asarray([float(np.mean(supplied[key])) for key in paired_keys])
    clip = np.asarray([stored[key] for key in paired_keys])
    n = len(paired_keys)
    print(f"Paired samples   : {n}")

    if n < MIN_PAIRS:
        fail(f"only {n} paired sample(s); at least {MIN_PAIRS} are needed.",
             "A correlation computed on a handful of images would be "
             "meaningless, so it is not printed.",
             f"Add ratings for at least {MIN_PAIRS} analysed designs.")

    if float(np.ptp(human)) == 0.0 or float(np.ptp(clip)) == 0.0:
        fail("every rating (or every CLIP score) is identical.",
             "Correlation is undefined when there is no variation.")

    rho = spearman(human, clip)
    r = _pearson(human, clip)
    rho_low, rho_high = bootstrap_ci(human, clip, spearman,
                                     args.bootstrap, args.seed)
    r_low, r_high = bootstrap_ci(human, clip, _pearson,
                                 args.bootstrap, args.seed)

    print()
    print(f"  Spearman rho   {rho:+.3f}   95% CI [{rho_low:+.3f}, {rho_high:+.3f}]")
    print(f"  Pearson r      {r:+.3f}   95% CI [{r_low:+.3f}, {r_high:+.3f}]")
    print(f"  (bootstrap: {args.bootstrap} resamples, seed {args.seed})")
    print()
    print("Interpretation guide - not a verdict:")
    print("  |rho| >= 0.70  strong agreement with these raters")
    print("  |rho| 0.40-0.70 moderate agreement")
    print("  |rho| < 0.40   weak agreement; treat CLIP as a rough "
          "second opinion only")
    print()
    print("This measures agreement with the people who supplied THIS "
          "file, on THIS set of images.")
    print("It does not validate the 0-100 Design Quality Score, which "
          "is rule-based and separate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Train the DesignScore aesthetic model.

Reads the designs an admin judged on the Train AI page, learns the link
between their 12 visual features and the quality score, then writes the
model to app/ai/aesthetic_model.pkl.

Usage:
    python scripts/train_model.py              # train on human labels only
    python scripts/train_model.py --augment    # also add synthetic variants
    python scripts/train_model.py --report     # print metrics, do not retrain
    python scripts/train_model.py --clear      # delete the trained model

Augmentation is optional and honest about what it is: it creates slightly
brighter / more contrasted / mirrored copies of designs already labelled.
Those copies do NOT add new knowledge - they only smooth out the model so
it behaves better between the examples you gave it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from app.ai import model_manager  # noqa: E402
from app.cv.feature_extraction import extract_features  # noqa: E402
from app.cv.preprocessing import load_rgb  # noqa: E402
from app.models import database  # noqa: E402


# =========================================================
# AUGMENTATION
# =========================================================

def _variants(path):
    """Three modest variations of one design."""
    rgb = load_rgb(path)
    out = [np.fliplr(rgb).copy()]                      # mirrored
    out.append(np.clip(rgb.astype(np.float32) * 1.12, 0, 255).astype(np.uint8))
    out.append(np.clip(rgb.astype(np.float32) * 0.88, 0, 255).astype(np.uint8))
    return out


def build_dataset(augment):
    """Turn the label table into model inputs.

    Returns (feature rows, target scores, synthetic row count, notes).
    """
    records = database.get_labelled_records()
    rows = []
    targets = []
    notes = []
    n_augmented = 0
    skipped = 0

    for rec in records:
        rows.append(json.loads(rec["features"]))
        targets.append(float(rec["score"]))

        if not augment:
            continue
        if not os.path.isfile(rec["image_path"]):
            skipped += 1
            continue
        try:
            for img in _variants(rec["image_path"]):
                rows.append(extract_features(img))
                targets.append(float(rec["score"]))
                n_augmented += 1
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop training
            notes.append("could not augment %s: %s" % (rec["image_path"], exc))

    if skipped:
        notes.append("%d labelled file(s) were missing on disk." % skipped)

    # Synthetic rows are copies of real ones. Keep real designs the majority
    # or the model would mostly be learning from duplicates.
    if augment and records and n_augmented >= len(records):
        keep = max(0, (len(records) - 1) - n_augmented)
        excess = n_augmented - keep
        if excess > 0:
            del rows[len(records) + keep:]
            del targets[len(records) + keep:]
            n_augmented -= excess
            notes.append(
                "trimmed %d synthetic row(s) so real designs stay the "
                "majority of the training set." % excess)

    return rows, targets, n_augmented, notes


# =========================================================
# REPORT
# =========================================================

def print_report(info):
    print()
    print("=" * 58)
    print("  DesignScore aesthetic model")
    print("=" * 58)
    print("  Trained at            : %s" % info.trained_at)
    print("  Training examples     : %d" % info.n_samples)
    print("    human labelled      : %d" % (info.n_samples - info.n_augmented))
    print("    synthetic variants  : %d" % info.n_augmented)
    print("  Class balance         : %d good / %d weak" % (info.n_good, info.n_weak))
    print("  Typical error (MAE)   : +/- %.2f points" % info.mae)
    print("  Pair ranking accuracy : %.1f%%" % (info.discrimination * 100))
    print()
    print("  Features the model relies on most:")
    for name, imp in list(info.feature_importance.items())[:6]:
        bar = "#" * int(imp * 60)
        print("    %-16s %5.1f%%  %s" % (name, imp * 100, bar))
    print("=" * 58)

    if info.n_samples < 20:
        print()
        print("  NOTE: this is a small training set. Treat the AI estimate as")
        print("  a rough second opinion, not a verdict. Label more designs to")
        print("  sharpen it.")
    print()


# =========================================================
# CLI
# =========================================================

def main():
    parser = argparse.ArgumentParser(description="Train the aesthetic model.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--augment", action="store_true",
                       help="add mirrored/brightness/contrast variants")
    group.add_argument("--report", action="store_true",
                       help="print the metrics of the existing model")
    group.add_argument("--clear", action="store_true",
                       help="delete the trained model")
    args = parser.parse_args()

    if args.clear:
        removed = model_manager.clear_model()
        print("Deleted the trained model." if removed
              else "No trained model to delete.")
        return 0

    if args.report:
        _, info = model_manager.load_model()
        if info is None:
            print("No model trained yet. Run: python scripts/train_model.py")
            return 1
        print_report(info)
        return 0

    rows, targets, n_augmented, notes = build_dataset(augment=args.augment)

    counts = database.get_label_counts()
    print("Labelled designs in the database: %d (%d good, %d weak)"
          % (counts["total"], counts["good"], counts["weak"]))
    if args.augment:
        print("Augmentation on: %d synthetic row(s) added" % n_augmented)
    else:
        print("Augmentation off (use --augment to add synthetic variants)")

    try:
        model, info = model_manager.train_model(rows, targets, n_augmented)
    except model_manager.TrainingDataError as exc:
        print()
        print("CANNOT TRAIN YET: %s" % exc)
        print()
        print("Next step: open /admin/label/ and judge more designs. Aim for a")
        print("roughly even split of Good and Weak.")
        return 2

    model_manager.save_model(model, info)

    for note in notes:
        print("  note: %s" % note)

    print_report(info)

    if info.discrimination < 0.7:
        print("  WARNING: the model separates good from weak designs only")
        print("  weakly. Label more, or re-label designs you were unsure")
        print("  about - noisy labels are the usual cause.")
        print()

    print("Saved: %s" % model_manager.MODEL_PATH)
    print("The AI estimate now appears on every assessment report.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
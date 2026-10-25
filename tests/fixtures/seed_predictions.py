"""
tests/fixtures/seed_predictions.py
────────────────────────────────────
Populates the Postgres ``predictions`` table with synthetic data for use in
CI pipelines and local integration testing of the drift monitor.

Two populations are generated to make drift detectable:
  • Reference-like  (70 % of rows) — brightness / contrast drawn from the
    training distribution (normal, centred around known NEU stats).
  • Drifted          (30 % of rows) — brightness shifted +40 points to
    simulate camera exposure change, and class distribution skewed toward
    "Crazing" to simulate concept drift.

This lets the drift_check.py CI job actually detect drift, making the
GitHub Issue alert testable end-to-end.

Usage
─────
  POSTGRES_DSN="postgresql://mlflow:mlflow@localhost:5432/mlflow" \\
    python tests/fixtures/seed_predictions.py

  # Seed only reference-like data (no drift)
  python tests/fixtures/seed_predictions.py --no-drift

  # Control how many rows to insert
  python tests/fixtures/seed_predictions.py --n-rows 200
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from datetime import datetime, timedelta, timezone

import numpy as np

POSTGRES_DSN = os.getenv(
    "POSTGRES_DSN",
    "postgresql://mlflow:mlflow@localhost:5432/mlflow",
)

CLASS_NAMES = [
    "Crazing", "Inclusion", "Patches",
    "Pitted_surface", "Rolled-in_scale", "Scratches",
]

# Approximate brightness/contrast stats from the NEU dataset (grayscale)
REF_BRIGHTNESS_MEAN,  REF_BRIGHTNESS_STD  = 118.0, 22.0
REF_CONTRAST_MEAN,    REF_CONTRAST_STD    = 45.0,  12.0
REF_CONFIDENCE_MEAN,  REF_CONFIDENCE_STD  = 0.87,  0.08

# Drifted distribution — brighter images (exposure drift) + skewed class dist
DRIFT_BRIGHTNESS_MEAN, DRIFT_BRIGHTNESS_STD = 158.0, 18.0   # +40 exposure shift
DRIFT_CONTRAST_MEAN,   DRIFT_CONTRAST_STD   = 30.0,  8.0    # lower contrast
DRIFT_CONFIDENCE_MEAN, DRIFT_CONFIDENCE_STD = 0.72,  0.12   # lower confidence

# Reference class distribution (roughly uniform across 6 classes)
REF_CLASS_PROBS   = [1/6] * 6

# Drifted class distribution (heavy on "Crazing")
DRIFT_CLASS_PROBS = [0.55, 0.10, 0.10, 0.10, 0.10, 0.05]


_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS predictions (
    id               SERIAL PRIMARY KEY,
    job_id           UUID        NOT NULL,
    camera_id        VARCHAR(64) NOT NULL DEFAULT '',
    image_path       TEXT        NOT NULL,
    predicted_class  VARCHAR(64) NOT NULL,
    class_index      INT         NOT NULL,
    confidence       DOUBLE PRECISION NOT NULL,
    probabilities    JSONB       NOT NULL DEFAULT '{}',
    inference_ms     DOUBLE PRECISION,
    captured_at      TIMESTAMPTZ,
    consumed_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    error            TEXT,
    CONSTRAINT predictions_job_id_uq UNIQUE (job_id)
);
"""


def _generate_rows(
    n_rows: int = 150,
    drift_fraction: float = 0.30,
    window_days: int = 7,
    rng: np.random.Generator = None,
) -> list[dict]:
    """Generate synthetic prediction rows."""
    if rng is None:
        rng = np.random.default_rng(42)

    n_drift   = int(n_rows * drift_fraction)
    n_nodrift = n_rows - n_drift

    now = datetime.now(timezone.utc)
    rows: list[dict] = []

    for i, (is_drift, count) in enumerate([
        (False, n_nodrift),
        (True,  n_drift),
    ]):
        if is_drift:
            brightness_arr = rng.normal(DRIFT_BRIGHTNESS_MEAN, DRIFT_BRIGHTNESS_STD, count).clip(0, 255)
            contrast_arr   = rng.normal(DRIFT_CONTRAST_MEAN,   DRIFT_CONTRAST_STD,   count).clip(0, 128)
            confidence_arr = rng.normal(DRIFT_CONFIDENCE_MEAN, DRIFT_CONFIDENCE_STD, count).clip(0.01, 0.99)
            class_indices  = rng.choice(len(CLASS_NAMES), size=count, p=DRIFT_CLASS_PROBS)
            camera_prefix  = "cam-drift"
        else:
            brightness_arr = rng.normal(REF_BRIGHTNESS_MEAN, REF_BRIGHTNESS_STD, count).clip(0, 255)
            contrast_arr   = rng.normal(REF_CONTRAST_MEAN,   REF_CONTRAST_STD,   count).clip(0, 128)
            confidence_arr = rng.normal(REF_CONFIDENCE_MEAN, REF_CONFIDENCE_STD, count).clip(0.01, 0.99)
            class_indices  = rng.choice(len(CLASS_NAMES), size=count, p=REF_CLASS_PROBS)
            camera_prefix  = "cam-main"

        for j in range(count):
            cls_idx       = int(class_indices[j])
            cls_name      = CLASS_NAMES[cls_idx]
            confidence    = float(confidence_arr[j])

            # Synthetic probability vector — concentrate on predicted class
            logits = rng.uniform(-2, 0, len(CLASS_NAMES))
            logits[cls_idx] += 3.0
            probs  = np.exp(logits) / np.exp(logits).sum()

            # Spread captured_at over the last `window_days`
            offset_s = rng.integers(0, window_days * 86_400)
            captured = now - timedelta(seconds=int(offset_s))

            rows.append({
                "job_id":          str(uuid.uuid4()),
                "camera_id":       f"{camera_prefix}-{(i * count + j) % 4 + 1:02d}",
                "image_path":      f"/data/raw/{cls_name}/ci_fixture_{j:04d}.bmp",
                "predicted_class": cls_name,
                "class_index":     cls_idx,
                "confidence":      round(confidence, 6),
                "probabilities":   json.dumps({CLASS_NAMES[k]: round(float(probs[k]), 6)
                                               for k in range(len(CLASS_NAMES))}),
                "inference_ms":    round(float(rng.uniform(10, 80)), 2),
                "captured_at":     captured.isoformat(),
                "consumed_at":     (captured + timedelta(milliseconds=float(rng.uniform(5, 50)))).isoformat(),
            })

    rng.shuffle(rows)  # mix drift and non-drift rows
    return rows


def seed(
    postgres_dsn: str = POSTGRES_DSN,
    n_rows: int = 150,
    drift_fraction: float = 0.30,
    clear_existing: bool = True,
) -> int:
    """
    Create the predictions table (if needed) and insert synthetic rows.

    Returns the number of rows inserted.
    """
    try:
        import psycopg2
        import psycopg2.extras
    except ImportError:
        print("ERROR: psycopg2-binary not installed.", file=sys.stderr)
        sys.exit(1)

    conn = psycopg2.connect(postgres_dsn)
    conn.autocommit = False

    try:
        with conn.cursor() as cur:
            # Ensure table exists
            cur.execute(_CREATE_TABLE_SQL)

            if clear_existing:
                cur.execute("TRUNCATE TABLE predictions RESTART IDENTITY;")
                print("Existing predictions cleared.")

        rows = _generate_rows(n_rows=n_rows, drift_fraction=drift_fraction)

        insert_sql = """
            INSERT INTO predictions
                (job_id, camera_id, image_path, predicted_class, class_index,
                 confidence, probabilities, inference_ms, captured_at, consumed_at)
            VALUES
                (%(job_id)s, %(camera_id)s, %(image_path)s, %(predicted_class)s,
                 %(class_index)s, %(confidence)s, %(probabilities)s, %(inference_ms)s,
                 %(captured_at)s, %(consumed_at)s)
            ON CONFLICT (job_id) DO NOTHING
        """

        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, insert_sql, rows, page_size=100)

        conn.commit()
        print(f"Inserted {len(rows)} synthetic predictions "
              f"({int(len(rows) * drift_fraction)} with drift, "
              f"{len(rows) - int(len(rows) * drift_fraction)} reference-like).")
        return len(rows)

    except Exception as exc:
        conn.rollback()
        print(f"ERROR during seeding: {exc}", file=sys.stderr)
        raise
    finally:
        conn.close()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Seed the predictions table with synthetic CI fixtures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--postgres-dsn", default=POSTGRES_DSN)
    p.add_argument("--n-rows", type=int, default=150,
                   help="Total rows to insert.")
    p.add_argument("--no-drift", action="store_true",
                   help="Generate only reference-like data (0% drift).")
    p.add_argument("--keep-existing", action="store_true",
                   help="Do not TRUNCATE before inserting.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    drift_frac = 0.0 if args.no_drift else 0.30
    seed(
        postgres_dsn=args.postgres_dsn,
        n_rows=args.n_rows,
        drift_fraction=drift_frac,
        clear_existing=not args.keep_existing,
    )

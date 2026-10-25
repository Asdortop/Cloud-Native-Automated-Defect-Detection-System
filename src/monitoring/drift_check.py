"""
src/monitoring/drift_check.py
──────────────────────────────
Data drift monitor for the NEU defect inspection ML pipeline.

Compares the statistical distribution of four features between:
  • Reference — features extracted from the training image set (data/raw/)
  • Current   — recent production predictions pulled from the Postgres
                predictions table (written by src/serving/consumer.py)

Features monitored
──────────────────
  brightness      Mean pixel value (grayscale)            — lighting / sensor drift
  contrast        Pixel standard deviation                 — focus / exposure drift
  predicted_class Model's predicted label distribution    — concept / label drift
  confidence      Prediction confidence score             — model calibration drift

Evidently library is used to:
  1. Run per-column statistical tests (Wasserstein for numeric, χ² for categorical)
  2. Generate a self-contained HTML report with interactive plots
  3. Return a machine-readable JSON summary for CI parsing

Exit codes
──────────
  0  No drift detected (all features within threshold)
  1  Drift detected (≥ threshold fraction of features drifted)
  2  Insufficient data  (fewer than --min-samples rows in current window)
  3  Configuration / connectivity error

Usage
─────
  # Production — read from Postgres, compute features from image files
  python src/monitoring/drift_check.py \\
      --training-dir  data/raw/ \\
      --postgres-dsn  "postgresql://mlflow:mlflow@localhost:5432/mlflow" \\
      --window-days   7 \\
      --threshold     0.3 \\
      --output-dir    reports/

  # CI — use pre-computed reference parquet + Postgres seeded with fixture
  python src/monitoring/drift_check.py \\
      --reference-parquet  data/reference_stats.parquet \\
      --postgres-dsn       "postgresql://mlflow:mlflow@localhost:5432/mlflow" \\
      --window-days        7

  # Save reference stats so CI can skip the slow image scan
  python src/monitoring/drift_check.py \\
      --training-dir       data/raw/ \\
      --save-reference     data/reference_stats.parquet \\
      --no-drift-check
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image

# ---------------------------------------------------------------------------
log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_POSTGRES_DSN:   str   = os.getenv("POSTGRES_DSN",  "postgresql://mlflow:mlflow@localhost:5432/mlflow")
DEFAULT_TRAINING_DIR:   str   = os.getenv("TRAINING_DIR",  "data/raw")
DEFAULT_WINDOW_DAYS:    int   = int(os.getenv("WINDOW_DAYS", "7"))
DEFAULT_MIN_SAMPLES:    int   = int(os.getenv("MIN_SAMPLES", "30"))
DEFAULT_DRIFT_THRESHOLD: float = float(os.getenv("DRIFT_THRESHOLD", "0.3"))
DEFAULT_OUTPUT_DIR:     str   = os.getenv("OUTPUT_DIR",    "reports")

# Image extensions to scan in the training directory
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

# Columns used for drift comparison
NUMERIC_FEATURES:      list[str] = ["brightness", "contrast", "confidence"]
CATEGORICAL_FEATURES:  list[str] = ["predicted_class"]
ALL_FEATURES:          list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# ===========================================================================
# Image feature extraction
# ===========================================================================

def extract_image_features(image_path: str) -> Optional[dict]:
    """
    Load a single image and compute brightness + contrast.

    Parameters
    ----------
    image_path : str  Local path or s3:// URI.

    Returns
    -------
    dict with keys ``brightness`` and ``contrast``, or ``None`` on failure.
    """
    try:
        if image_path.startswith("s3://"):
            raw_bytes = _fetch_from_s3(image_path)
            from io import BytesIO
            img = Image.open(BytesIO(raw_bytes)).convert("L")
        else:
            img = Image.open(image_path).convert("L")

        arr = np.array(img, dtype=np.float32)
        return {
            "brightness": float(arr.mean()),
            "contrast":   float(arr.std()),
        }
    except Exception as exc:
        log.debug("Could not extract features from %s: %s", image_path, exc)
        return None


def _fetch_from_s3(s3_uri: str) -> bytes:
    """Download bytes from MinIO / S3 using boto3."""
    import boto3
    from botocore.client import Config

    without_scheme = s3_uri[len("s3://"):]
    bucket, _, key  = without_scheme.partition("/")

    s3 = boto3.client(
        "s3",
        endpoint_url=os.getenv("MINIO_ENDPOINT", "http://localhost:9000"),
        aws_access_key_id=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
        aws_secret_access_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )
    return s3.get_object(Bucket=bucket, Key=key)["Body"].read()


# ===========================================================================
# Data loading
# ===========================================================================

def load_training_reference(training_dir: str) -> pd.DataFrame:
    """
    Walk the training directory tree (class_name/image.ext) and extract
    brightness, contrast, and class label for each image.

    Expects a directory structure like::

        data/raw/
        ├── Crazing/
        │   ├── img_001.bmp
        │   └── ...
        ├── Inclusion/
        └── ...

    Returns
    -------
    pd.DataFrame with columns: brightness, contrast, predicted_class, confidence
    """
    root = Path(training_dir)
    if not root.is_dir():
        raise ValueError(f"Training directory not found: {root.resolve()}")

    records: list[dict] = []
    class_dirs = [d for d in root.iterdir() if d.is_dir()]
    if not class_dirs:
        raise ValueError(f"No class subdirectories found under: {root}")

    log.info("Scanning %d class directories under: %s", len(class_dirs), root)

    for class_dir in sorted(class_dirs):
        class_name = class_dir.name
        image_paths = [
            p for p in class_dir.rglob("*")
            if p.suffix.lower() in IMAGE_EXTENSIONS
        ]
        log.info("  %s: %d images", class_name, len(image_paths))

        for img_path in image_paths:
            feats = extract_image_features(str(img_path))
            if feats is not None:
                records.append({
                    **feats,
                    "predicted_class": class_name,
                    "confidence":      1.0,   # ground-truth ≡ 100% confidence
                })

    df = pd.DataFrame(records)
    log.info(
        "Reference dataset: %d samples across %d classes.",
        len(df), df["predicted_class"].nunique(),
    )
    return df


def load_production_data(
    postgres_dsn: str,
    window_days: int = DEFAULT_WINDOW_DAYS,
    compute_image_features: bool = True,
) -> pd.DataFrame:
    """
    Pull recent predictions from the Postgres ``predictions`` table and
    optionally compute per-image brightness / contrast.

    Parameters
    ----------
    postgres_dsn : str
        SQLAlchemy-compatible Postgres DSN.
    window_days : int
        How many days back to pull records.
    compute_image_features : bool
        When True, try to load each image from ``image_path`` and compute
        brightness / contrast.  Set False to skip image I/O (faster in CI
        when images are seeded directly into the DB as extra columns).

    Returns
    -------
    pd.DataFrame with columns: brightness, contrast, predicted_class, confidence
    """
    try:
        import psycopg2
        import psycopg2.extras
    except ImportError as exc:
        raise RuntimeError("psycopg2-binary is required: pip install psycopg2-binary") from exc

    log.info(
        "Connecting to Postgres and pulling last %d days of predictions …",
        window_days,
    )

    conn = psycopg2.connect(postgres_dsn)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT image_path,
                       predicted_class,
                       confidence,
                       -- pull pre-computed stats if the columns exist
                       (SELECT column_name FROM information_schema.columns
                        WHERE table_name = 'predictions'
                          AND column_name = 'brightness'
                        LIMIT 1) IS NOT NULL AS has_precomputed,
                       CASE WHEN EXISTS (
                           SELECT 1 FROM information_schema.columns
                           WHERE table_name='predictions' AND column_name='brightness')
                       THEN brightness ELSE NULL END AS brightness,
                       CASE WHEN EXISTS (
                           SELECT 1 FROM information_schema.columns
                           WHERE table_name='predictions' AND column_name='contrast')
                       THEN contrast ELSE NULL END AS contrast
                FROM   predictions
                WHERE  consumed_at >= NOW() - INTERVAL %s
                ORDER  BY consumed_at DESC
                """,
                (f"{window_days} days",),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        log.warning("No production predictions found in the last %d days.", window_days)
        return pd.DataFrame(columns=ALL_FEATURES)

    df = pd.DataFrame(rows)
    log.info("Pulled %d production records.", len(df))

    # ── Compute image features if not pre-computed in the DB ───────────────
    if compute_image_features and (df.get("brightness") is None or df["brightness"].isna().all()):
        log.info("Computing image features from image_path …")
        features = []
        missing = 0
        for path in df["image_path"]:
            feats = extract_image_features(str(path))
            if feats:
                features.append(feats)
            else:
                features.append({"brightness": np.nan, "contrast": np.nan})
                missing += 1

        feat_df = pd.DataFrame(features)
        df["brightness"] = feat_df["brightness"].values
        df["contrast"]   = feat_df["contrast"].values

        if missing > 0:
            pct = 100 * missing / len(df)
            log.warning(
                "%d / %d images (%.0f%%) could not be loaded from disk — "
                "brightness/contrast will be NaN for those rows.",
                missing, len(df), pct,
            )

    # Keep only the columns we care about
    result_cols = [c for c in ALL_FEATURES if c in df.columns]
    return df[result_cols].copy()


def load_reference_from_parquet(parquet_path: str) -> pd.DataFrame:
    """Load pre-computed reference stats from a Parquet file."""
    path = Path(parquet_path)
    if not path.exists():
        raise FileNotFoundError(f"Reference parquet not found: {path.resolve()}")
    df = pd.read_parquet(path)
    log.info("Loaded %d reference samples from: %s", len(df), parquet_path)
    return df


# ===========================================================================
# Drift detection
# ===========================================================================

def run_drift_check(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    drift_threshold: float,
    output_dir: Path,
    report_name: str,
) -> dict:
    """
    Run Evidently drift analysis and save the HTML + JSON summary.

    Parameters
    ----------
    reference_df : pd.DataFrame   Training-set feature statistics.
    current_df   : pd.DataFrame   Recent production feature statistics.
    drift_threshold : float       Share of features that must drift to declare
                                  dataset drift (0.0 – 1.0).
    output_dir : Path             Directory for report files.
    report_name : str             Base filename (without extension).

    Returns
    -------
    dict  Machine-readable summary suitable for CI parsing.
    """
    try:
        try:
            from evidently.metric_preset import DataDriftPreset
            from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric, DatasetSummaryMetric
            from evidently.report import Report
            from evidently import ColumnMapping
        except ImportError:
            from evidently.legacy.metric_preset import DataDriftPreset
            from evidently.legacy.metrics import ColumnDriftMetric, DatasetDriftMetric, DatasetSummaryMetric
            from evidently.legacy.report import Report
            from evidently.legacy.pipeline.column_mapping import ColumnMapping
    except ImportError as exc:
        raise RuntimeError(
            "evidently is required: pip install evidently"
        ) from exc

    # ── Align columns — only compare features present in both DataFrames ───
    shared_numeric     = [c for c in NUMERIC_FEATURES     if c in reference_df.columns and c in current_df.columns]
    shared_categorical = [c for c in CATEGORICAL_FEATURES if c in reference_df.columns and c in current_df.columns]
    all_shared = shared_numeric + shared_categorical

    if not all_shared:
        raise ValueError(
            "No common feature columns found between reference and current datasets."
        )

    ref = reference_df[all_shared].copy()
    cur = current_df[all_shared].copy()

    # Drop rows where ALL feature values are NaN
    ref = ref.dropna(how="all")
    cur = cur.dropna(how="all")

    log.info(
        "Running Evidently drift check: %d reference vs %d current samples, "
        "features=%s, threshold=%.0f%%",
        len(ref), len(cur), all_shared, drift_threshold * 100,
    )

    # ── Column mapping ──────────────────────────────────────────────────────
    column_mapping = ColumnMapping(
        numerical_features=shared_numeric,
        categorical_features=shared_categorical,
    )

    # ── Build report ────────────────────────────────────────────────────────
    metrics = [
        DatasetSummaryMetric(),
        DatasetDriftMetric(drift_share=drift_threshold),
    ]
    for col in all_shared:
        metrics.append(ColumnDriftMetric(column_name=col))

    report = Report(metrics=metrics)
    report.run(
        reference_data=ref,
        current_data=cur,
        column_mapping=column_mapping,
    )

    # ── Persist HTML report ─────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / f"{report_name}.html"
    report.save_html(str(html_path))
    log.info("HTML drift report saved: %s", html_path)

    # ── Parse results ───────────────────────────────────────────────────────
    result_dict = json.loads(report.json())
    summary = _parse_evidently_results(
        result_dict=result_dict,
        html_path=html_path,
        reference_n=len(ref),
        current_n=len(cur),
        drift_threshold=drift_threshold,
        feature_cols=all_shared,
    )

    # ── Persist JSON summary ────────────────────────────────────────────────
    json_path = output_dir / f"{report_name}.json"
    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    log.info("JSON summary saved: %s", json_path)

    return summary


def _parse_evidently_results(
    result_dict: dict,
    html_path: Path,
    reference_n: int,
    current_n: int,
    drift_threshold: float,
    feature_cols: list[str],
) -> dict:
    """
    Extract key drift metrics from the Evidently result dict into a
    flat structure that's easy to consume in CI scripts / GitHub Script.
    """
    dataset_drift = False
    drift_share   = 0.0
    n_drifted     = 0
    column_results: dict[str, dict] = {}

    for metric in result_dict.get("metrics", []):
        metric_type = metric.get("metric", "")
        result      = metric.get("result", {})

        if "DatasetDriftMetric" in metric_type:
            dataset_drift = result.get("dataset_drift", False)
            n_drifted     = result.get("n_drifted_features", result.get("number_of_drifted_columns", 0))
            total_features = result.get("n_features", result.get("number_of_columns", len(feature_cols)))
            drift_share   = n_drifted / max(total_features, 1)

        elif "ColumnDriftMetric" in metric_type:
            col = result.get("column_name", "")
            column_results[col] = {
                "drift_detected": result.get("drift_detected", False),
                "drift_score":    round(float(result.get("drift_score", 0.0)), 6),
                "stattest":       result.get("stattest_name", result.get("stattest", "unknown")),
                "threshold":      result.get("threshold", 0.0),
                "p_value":        result.get("p_value"),
            }

    drifted_features = [col for col, r in column_results.items() if r["drift_detected"]]

    return {
        "drift_detected":  dataset_drift,
        "drift_share":     round(drift_share, 4),
        "drift_threshold": drift_threshold,
        "n_drifted":       n_drifted,
        "n_features":      len(feature_cols),
        "drifted_features": drifted_features,
        "column_results":  column_results,
        "n_reference_samples": reference_n,
        "n_current_samples":   current_n,
        "html_report":     str(html_path),
        "timestamp":       datetime.now(timezone.utc).isoformat(),
    }


# ===========================================================================
# CLI
# ===========================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Check for data and concept drift between training and recent "
            "production predictions.  Exit 0 = clean, 1 = drift, 2 = insufficient data."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Reference source (mutually exclusive) ─────────────────────────────
    ref_group = p.add_mutually_exclusive_group()
    ref_group.add_argument(
        "--training-dir",
        default=DEFAULT_TRAINING_DIR,
        help="Root directory of training images (class_name/image.ext).",
    )
    ref_group.add_argument(
        "--reference-parquet",
        help="Pre-computed reference feature stats (.parquet). "
             "Faster than --training-dir when images are unavailable.",
    )

    p.add_argument(
        "--save-reference",
        help="Save extracted reference feature stats to this parquet path.",
    )
    p.add_argument(
        "--no-drift-check",
        action="store_true",
        help="Only extract and save reference stats; skip the drift report.",
    )

    # ── Current (production) source ────────────────────────────────────────
    p.add_argument(
        "--postgres-dsn",
        default=DEFAULT_POSTGRES_DSN,
        help="Postgres connection string for the predictions table.",
    )
    p.add_argument(
        "--window-days",
        type=int,
        default=DEFAULT_WINDOW_DAYS,
        help="Number of days of production data to include in 'current'.",
    )
    p.add_argument(
        "--no-image-load",
        action="store_true",
        help="Skip loading images to compute brightness/contrast. Use only "
             "predicted_class and confidence for drift detection.",
    )

    # ── Thresholds ─────────────────────────────────────────────────────────
    p.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_DRIFT_THRESHOLD,
        help="Fraction of features that must drift to declare dataset drift (0.0–1.0).",
    )
    p.add_argument(
        "--min-samples",
        type=int,
        default=DEFAULT_MIN_SAMPLES,
        help="Minimum production rows required; exit 2 if fewer.",
    )

    # ── Output ─────────────────────────────────────────────────────────────
    p.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for HTML report and JSON summary.",
    )
    p.add_argument(
        "--report-name",
        default=None,
        help="Base filename for report files. Defaults to drift_report_<date>.",
    )

    return p.parse_args()


def main() -> int:
    args = _parse_args()

    report_name = args.report_name or (
        f"drift_report_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir = Path(args.output_dir)

    # ── 1. Load reference data ─────────────────────────────────────────────
    if args.reference_parquet:
        reference_df = load_reference_from_parquet(args.reference_parquet)
    else:
        try:
            reference_df = load_training_reference(args.training_dir)
        except (ValueError, FileNotFoundError) as exc:
            log.error("Could not load training reference: %s", exc)
            return 3

    if reference_df.empty:
        log.error("Reference dataset is empty.")
        return 3

    # ── Optionally save reference parquet ─────────────────────────────────
    if args.save_reference:
        save_path = Path(args.save_reference)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        reference_df.to_parquet(save_path, index=False)
        log.info("Reference stats saved to: %s (%d rows)", save_path, len(reference_df))

    if args.no_drift_check:
        log.info("--no-drift-check set; skipping drift analysis.")
        return 0

    # ── 2. Load production (current) data ──────────────────────────────────
    try:
        current_df = load_production_data(
            postgres_dsn=args.postgres_dsn,
            window_days=args.window_days,
            compute_image_features=not args.no_image_load,
        )
    except Exception as exc:
        log.error("Could not load production data: %s", exc)
        return 3

    # ── 3. Guard: minimum sample check ────────────────────────────────────
    if len(current_df) < args.min_samples:
        log.warning(
            "Only %d production samples found (minimum: %d). "
            "Statistical tests will be unreliable — exiting with code 2.",
            len(current_df), args.min_samples,
        )
        # Write a minimal summary so CI can still parse the output
        summary = {
            "drift_detected":      False,
            "insufficient_data":   True,
            "n_current_samples":   len(current_df),
            "min_samples":         args.min_samples,
            "n_reference_samples": len(reference_df),
            "timestamp":           datetime.now(timezone.utc).isoformat(),
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / f"{report_name}.json", "w") as fh:
            json.dump(summary, fh, indent=2)
        return 2

    # ── 4. Run drift check ─────────────────────────────────────────────────
    try:
        summary = run_drift_check(
            reference_df=reference_df,
            current_df=current_df,
            drift_threshold=args.threshold,
            output_dir=output_dir,
            report_name=report_name,
        )
    except Exception as exc:
        log.error("Drift check failed: %s", exc, exc_info=True)
        return 3

    # ── 5. Print human-readable summary ───────────────────────────────────
    _print_summary(summary)

    return 1 if summary["drift_detected"] else 0


def _print_summary(summary: dict) -> None:
    """Print a formatted drift summary to stdout."""
    icon  = "🔴" if summary["drift_detected"] else "🟢"
    label = "DRIFT DETECTED" if summary["drift_detected"] else "NO DRIFT"

    print()
    print(f"{'─'*60}")
    print(f"  {icon}  {label}")
    print(f"{'─'*60}")
    print(f"  Reference samples : {summary.get('n_reference_samples', '?'):>6}")
    print(f"  Current samples   : {summary.get('n_current_samples', '?'):>6}")
    print(f"  Drift share       : {summary.get('drift_share', 0):.0%}  (threshold: {summary.get('drift_threshold', 0):.0%})")
    print(f"  Drifted features  : {', '.join(summary.get('drifted_features') or ['none'])}")
    print()

    col_results = summary.get("column_results", {})
    if col_results:
        print(f"  {'Feature':<20} {'Drifted?':<10} {'Score':<10} {'Test'}")
        print(f"  {'─'*20} {'─'*10} {'─'*10} {'─'*20}")
        for col, res in col_results.items():
            drifted = "YES ⚠" if res["drift_detected"] else "no"
            print(
                f"  {col:<20} {drifted:<10} {res['drift_score']:<10.4f} {res['stattest']}"
            )
    print()
    print(f"  HTML report: {summary.get('html_report', 'N/A')}")
    print(f"{'─'*60}")
    print()


if __name__ == "__main__":
    sys.exit(main())

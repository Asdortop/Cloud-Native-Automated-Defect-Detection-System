"""
src/training/evaluate_gate.py
------------------------------
Quality gate that decides whether a freshly-trained model should advance
to the MLflow Model Registry's "Staging" stage.

Gate logic (both conditions must pass)
---------------------------------------
1. Absolute threshold  – every class recall AND macro recall must be >= THRESHOLD
2. Relative champion   – macro recall must be >= the currently "Production" model's
                         macro recall (no regression allowed)

If both pass → register the run's model artefact as a new Model Registry version
              → transition that version to "Staging"
If either fails → print a clear report and exit with code 1 (CI-friendly)

Usage
-----
    python src/training/evaluate_gate.py \\
        --model-name defect-efficientnet_b0 \\
        --experiment  defect-inspection \\
        --threshold   0.90 \\
        [--run-id <specific-run-id>]      # defaults to latest finished run
        [--tracking-uri http://localhost:5000]
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Optional

import mlflow
from mlflow import MlflowClient
from mlflow.entities import Run
from mlflow.entities.model_registry import ModelVersion

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# Default quality gate threshold
DEFAULT_THRESHOLD = 0.90

# Metric key written by train.py for macro recall on the *validation* set
MACRO_RECALL_KEY = "val_recall_macro"

# Prefix shared by per-class recall metrics logged during training
CLASS_RECALL_PREFIX = "val_recall_"


# ===========================================================================
# Helpers
# ===========================================================================

def get_latest_run(
    client: MlflowClient,
    experiment_name: str,
    run_id: Optional[str] = None,
) -> Run:
    """
    Return a specific run (if *run_id* is given) or the most-recently
    finished run in *experiment_name*.
    """
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(
            f"Experiment '{experiment_name}' not found in the tracking server. "
            "Have you run at least one training job?"
        )

    if run_id:
        run = client.get_run(run_id)
        log.info("Using specified run: %s", run.info.run_id)
        return run

    # Search for the latest FINISHED run, ordered by start time descending
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError(
            f"No finished runs found in experiment '{experiment_name}'."
        )
    run = runs[0]
    log.info(
        "Latest finished run: %s (started %s)",
        run.info.run_id,
        run.info.start_time,
    )
    return run


def get_recall_metrics(run: Run, prefix: str = CLASS_RECALL_PREFIX) -> dict[str, float]:
    """
    Extract all per-class and macro recall metrics from a run's metric dict.

    Returns a dict like:
        { 'val_recall_macro': 0.93,
          'val_recall_Crazing': 0.91,
          'val_recall_Inclusion': 0.94, ... }
    """
    metrics = {
        k: v
        for k, v in run.data.metrics.items()
        if k.startswith(prefix) or k == MACRO_RECALL_KEY
    }
    if not metrics:
        raise RuntimeError(
            f"Run {run.info.run_id} has no recall metrics matching "
            f"prefix '{prefix}'. Was the model trained with train.py?"
        )
    return metrics


def get_production_model(
    client: MlflowClient,
    model_name: str,
) -> Optional[ModelVersion]:
    """
    Return the ModelVersion currently tagged as 'Production', or None if
    no production model exists yet.

    Supports both the legacy stage-based API and the modern alias-based API.
    """
    # ── Try alias-based (MLflow ≥ 2.9 recommended approach) ──────────────
    try:
        mv = client.get_model_version_by_alias(model_name, "production")
        log.info(
            "Champion (alias='production'): %s v%s (recall=?)",
            model_name, mv.version,
        )
        return mv
    except Exception:
        pass

    # ── Fall back to stage-based (legacy, still supported in MLflow 3.x) ──
    try:
        versions = client.get_latest_versions(model_name, stages=["Production"])
        if versions:
            mv = versions[0]
            log.info(
                "Champion (stage='Production'): %s v%s",
                model_name, mv.version,
            )
            return mv
    except Exception:
        pass

    log.info(
        "No production model found for '%s'. Threshold-only gate will apply.",
        model_name,
    )
    return None


def get_run_metric_for_version(
    client: MlflowClient,
    model_version: ModelVersion,
) -> Optional[float]:
    """Retrieve val_recall_macro for the run that produced *model_version*."""
    try:
        source_run = client.get_run(model_version.run_id)
        return source_run.data.metrics.get(MACRO_RECALL_KEY)
    except Exception as exc:
        log.warning("Could not fetch metrics for production model run: %s", exc)
        return None


def register_and_promote(
    client: MlflowClient,
    run: Run,
    model_name: str,
) -> ModelVersion:
    """
    Register the model artefact from *run* and transition it to 'Staging'.

    Uses the modern alias API first; falls back to legacy stage transition.
    """
    artifact_uri = f"runs:/{run.info.run_id}/model"
    log.info("Registering model from: %s", artifact_uri)

    mv = mlflow.register_model(artifact_uri, model_name)
    log.info("Registered %s v%s", model_name, mv.version)

    # ── Set alias (modern) ────────────────────────────────────────────────
    try:
        client.set_registered_model_alias(model_name, "staging", mv.version)
        log.info("Set alias 'staging' → %s v%s", model_name, mv.version)
    except Exception:
        # ── Fallback: stage transition (legacy) ───────────────────────────
        try:
            client.transition_model_version_stage(
                name=model_name,
                version=mv.version,
                stage="Staging",
                archive_existing_versions=False,
            )
            log.info(
                "Transitioned %s v%s → Staging (legacy API)",
                model_name, mv.version,
            )
        except Exception as exc2:
            log.error("Could not set Staging via alias or stage API: %s", exc2)

    return mv


# ===========================================================================
# Gate evaluation
# ===========================================================================

def evaluate_gate(
    model_name: str,
    experiment_name: str,
    threshold: float = DEFAULT_THRESHOLD,
    run_id: Optional[str] = None,
    tracking_uri: str = "http://localhost:5000",
) -> bool:
    """
    Run the full quality gate.

    Returns
    -------
    bool
        ``True`` if the gate passes and the model was promoted to Staging.
        ``False`` if the gate fails.
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    # ── 1. Load candidate run ─────────────────────────────────────────────
    run = get_latest_run(client, experiment_name, run_id)
    recall_metrics = get_recall_metrics(run)
    candidate_macro = recall_metrics.get(MACRO_RECALL_KEY, 0.0)

    log.info("=" * 60)
    log.info("Candidate run   : %s", run.info.run_id)
    log.info("Recall metrics  :")
    for k, v in sorted(recall_metrics.items()):
        status = "✓" if v >= threshold else "✗"
        log.info("  %s  %s = %.4f  (threshold %.2f)", status, k, v, threshold)

    # ── 2. Absolute threshold check ───────────────────────────────────────
    failed_metrics = {k: v for k, v in recall_metrics.items() if v < threshold}
    gate_1_pass = len(failed_metrics) == 0

    if gate_1_pass:
        log.info("Gate 1 (absolute threshold %.2f): PASS ✓", threshold)
    else:
        log.warning("Gate 1 (absolute threshold %.2f): FAIL ✗", threshold)
        for k, v in failed_metrics.items():
            log.warning("  %s = %.4f < %.2f", k, v, threshold)

    # ── 3. Champion comparison ────────────────────────────────────────────
    champion_mv = get_production_model(client, model_name)
    gate_2_pass = True

    if champion_mv is not None:
        champion_macro = get_run_metric_for_version(client, champion_mv)
        if champion_macro is not None:
            gate_2_pass = candidate_macro >= champion_macro
            symbol = "✓" if gate_2_pass else "✗"
            log.info(
                "Gate 2 (vs champion v%s): %s  candidate=%.4f  champion=%.4f",
                champion_mv.version, symbol, candidate_macro, champion_macro,
            )
            if not gate_2_pass:
                log.warning(
                    "Gate 2 FAIL: candidate recall (%.4f) is below champion (%.4f).",
                    candidate_macro, champion_macro,
                )
        else:
            log.info(
                "Gate 2: champion recall metric unavailable — skipping comparison."
            )
    else:
        log.info("Gate 2: no production model — skipping champion comparison.")

    # ── 4. Decision ──────────────────────────────────────────────────────
    log.info("=" * 60)
    all_pass = gate_1_pass and gate_2_pass

    if all_pass:
        log.info("GATE PASSED — promoting model to Staging.")
        mv = register_and_promote(client, run, model_name)
        log.info(
            "Model '%s' version %s is now in Staging.",
            model_name, mv.version,
        )
        return True
    else:
        log.error("GATE FAILED — model will NOT be promoted.")
        return False


# ===========================================================================
# CLI
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Quality gate: promote a model to Staging if recall thresholds pass.",
    )
    p.add_argument(
        "--model-name",
        default="defect-efficientnet_b0",
        help="Registered model name in the MLflow Model Registry.",
    )
    p.add_argument(
        "--experiment",
        default="defect-inspection",
        help="MLflow experiment name to search for the latest run.",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Minimum recall (per-class and macro) required (default: {DEFAULT_THRESHOLD}).",
    )
    p.add_argument(
        "--run-id",
        default=None,
        help="Evaluate a specific run ID instead of the latest finished run.",
    )
    p.add_argument(
        "--tracking-uri",
        default="http://localhost:5000",
        help="MLflow tracking server URI.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    passed = evaluate_gate(
        model_name=args.model_name,
        experiment_name=args.experiment,
        threshold=args.threshold,
        run_id=args.run_id,
        tracking_uri=args.tracking_uri,
    )
    sys.exit(0 if passed else 1)

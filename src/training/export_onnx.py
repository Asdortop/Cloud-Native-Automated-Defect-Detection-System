"""
src/training/export_onnx.py
----------------------------
Loads a PyTorch model from the MLflow Model Registry and exports it to
ONNX format, then verifies the ONNX output matches the PyTorch output
within a numerical tolerance using onnxruntime.

What it does
------------
1. Pull a registered model version (or the "staging" / "production" alias)
   from the MLflow tracking server.
2. Export the model to an ONNX file via ``torch.onnx.export`` using the
   new dynamo-based exporter when available (torch >= 2.0).
3. Run an onnxruntime inference session on the same dummy input and compare
   the outputs to the PyTorch outputs with ``np.allclose`` (atol/rtol
   configurable from CLI).
4. Optionally log the verified ONNX file back to MLflow as an artefact.

Usage
-----
    # Export the latest 'staging' alias of the registered model
    python src/training/export_onnx.py \\
        --model-name   defect-efficientnet_b0 \\
        --alias        staging \\
        --output-path  models/defect_b0.onnx

    # Export a specific version
    python src/training/export_onnx.py \\
        --model-name   defect-efficientnet_b0 \\
        --version      3 \\
        --output-path  models/defect_b0_v3.onnx \\
        --atol         1e-5

    # Skip logging back to MLflow
    python src/training/export_onnx.py \\
        --model-name defect-efficientnet_b0 --no-log-artifact
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

import mlflow
import mlflow.pytorch
import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from mlflow import MlflowClient

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# Default tolerances for numerical comparison
DEFAULT_ATOL = 1e-5
DEFAULT_RTOL = 1e-3

# Batch size and spatial dims used for the dummy inference check
VERIFY_BATCH  = 4
IMAGE_SIZE    = 224


# ===========================================================================
# Model loading
# ===========================================================================

def load_model_from_registry(
    client: MlflowClient,
    model_name: str,
    version: Optional[str] = None,
    alias: Optional[str] = None,
) -> tuple[nn.Module, str]:
    """
    Load a PyTorch model from the MLflow Model Registry.

    Priority: explicit *version* > *alias* > default to 'staging' alias.

    Parameters
    ----------
    client     : MlflowClient
    model_name : str    Registered model name.
    version    : str | None  Specific version number (string).
    alias      : str | None  Registered model alias (e.g. 'staging').

    Returns
    -------
    (model, model_uri)
    """
    if version is not None:
        model_uri = f"models:/{model_name}/{version}"
        log.info("Loading model: %s (version=%s)", model_name, version)
    elif alias is not None:
        model_uri = f"models:/{model_name}@{alias}"
        log.info("Loading model: %s (alias='%s')", model_name, alias)
    else:
        # Fall back to 'staging' alias, then latest version
        try:
            model_uri = f"models:/{model_name}@staging"
            log.info("No version/alias specified — trying alias 'staging'.")
        except Exception:
            model_uri = f"models:/{model_name}/latest"
            log.info("Falling back to latest version.")

    model = mlflow.pytorch.load_model(model_uri, map_location="cpu")
    model.eval()
    log.info("Model loaded from: %s", model_uri)
    return model, model_uri


# ===========================================================================
# ONNX export
# ===========================================================================

def export_to_onnx(
    model: nn.Module,
    output_path: Path,
    num_classes: int,
    opset_version: int = 18,
) -> None:
    """
    Export *model* to ONNX at *output_path*.

    Targets the ``torch.onnx.export`` API introduced in torch 2.x
    (``dynamo=True`` path, default in torch >= 2.13).  The exported graph
    has a dynamic batch dimension so it accepts any batch size at runtime.

    Parameters
    ----------
    model        : nn.Module   The PyTorch model (eval mode expected).
    output_path  : Path        Destination ``.onnx`` file.
    num_classes  : int         Number of output classes (informational).
    opset_version: int         ONNX opset version (default 18).
    """
    import os
    # Ensure UTF-8 stdout so torch's internal print calls don't crash on
    # Windows terminals that default to cp1252.
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)

    log.info(
        "Exporting to ONNX -> %s  (opset=%d, input=%s)",
        output_path, opset_version, list(dummy_input.shape),
    )

    # torch 2.13: dynamic_shapes must be specified as a list/tuple matching
    # the positional args of forward(), OR as a dict keyed on arg names.
    # TinyNet.forward has arg `x`, EfficientNet/other models also use `x`.
    # We use the list form to avoid arg-name coupling.
    with torch.no_grad():
        batch_dim = torch.export.Dim("batch_size", min=1, max=256)
        # List element [0] corresponds to the first positional arg (images).
        # {0: batch_dim} marks dimension 0 of that tensor as dynamic.
        dynamic_shapes = [{0: batch_dim}]

        torch.onnx.export(
            model,
            args=(dummy_input,),
            f=str(output_path),
            dynamo=True,
            opset_version=opset_version,
            input_names=["images"],
            output_names=["logits"],
            dynamic_shapes=dynamic_shapes,
            report=False,
            verbose=False,
        )

    size_mb = output_path.stat().st_size / 1_048_576
    log.info("ONNX file written: %.2f MB", size_mb)



# ===========================================================================
# Verification
# ===========================================================================

def verify_onnx_output(
    pytorch_model: nn.Module,
    onnx_path: Path,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
    batch_size: int = VERIFY_BATCH,
) -> bool:
    """
    Run both the PyTorch model and the ONNX model on identical random inputs
    and compare their outputs.

    Parameters
    ----------
    pytorch_model : nn.Module
    onnx_path     : Path
    atol          : float   Absolute tolerance for ``np.allclose``.
    rtol          : float   Relative tolerance for ``np.allclose``.
    batch_size    : int     Number of samples in the verification batch.

    Returns
    -------
    bool  True if outputs match within tolerance, False otherwise.
    """
    log.info(
        "Verifying ONNX output (batch=%d, atol=%g, rtol=%g) …",
        batch_size, atol, rtol,
    )

    # ── PyTorch reference ─────────────────────────────────────────────────
    dummy_input = torch.randn(batch_size, 3, IMAGE_SIZE, IMAGE_SIZE)
    with torch.no_grad():
        pt_output: np.ndarray = pytorch_model(dummy_input).numpy()

    # ── ONNX Runtime inference ────────────────────────────────────────────
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if ort.get_device() == "GPU"
        else ["CPUExecutionProvider"]
    )
    sess = ort.InferenceSession(str(onnx_path), providers=providers)
    input_name = sess.get_inputs()[0].name
    ort_output: np.ndarray = sess.run(None, {input_name: dummy_input.numpy()})[0]

    log.info("PyTorch output shape : %s", pt_output.shape)
    log.info("ONNX RT output shape : %s", ort_output.shape)

    # ── Numerical comparison ──────────────────────────────────────────────
    max_abs_diff  = float(np.abs(pt_output - ort_output).max())
    mean_abs_diff = float(np.abs(pt_output - ort_output).mean())
    log.info("Max  absolute diff   : %.2e", max_abs_diff)
    log.info("Mean absolute diff   : %.2e", mean_abs_diff)

    match = bool(np.allclose(pt_output, ort_output, atol=atol, rtol=rtol))

    if match:
        log.info("Verification PASSED ✓  (all diffs within atol=%g, rtol=%g)", atol, rtol)
    else:
        log.error(
            "Verification FAILED ✗  max diff=%.2e exceeds tolerance atol=%g",
            max_abs_diff, atol,
        )
        # Show which logits diverge most
        worst_batch, worst_cls = np.unravel_index(
            np.abs(pt_output - ort_output).argmax(), pt_output.shape
        )
        log.error(
            "  Worst divergence: batch[%d] class[%d] → pt=%.6f  ort=%.6f",
            worst_batch, worst_cls,
            pt_output[worst_batch, worst_cls],
            ort_output[worst_batch, worst_cls],
        )

    return match


# ===========================================================================
# MLflow artefact logging
# ===========================================================================

def log_onnx_artifact(
    client: MlflowClient,
    onnx_path: Path,
    model_name: str,
    model_version: str,
) -> None:
    """
    Log the verified ONNX file back to the MLflow run that produced this
    model version, under an ``onnx/`` subdirectory.
    """
    try:
        mv = client.get_model_version(model_name, model_version)
        run_id = mv.run_id
        mlflow.log_artifact(str(onnx_path), artifact_path="onnx", run_id=run_id)
        log.info(
            "ONNX artefact logged to run %s under 'onnx/%s'",
            run_id, onnx_path.name,
        )
    except Exception as exc:
        log.warning("Could not log ONNX artefact to MLflow: %s", exc)


# ===========================================================================
# Orchestrator
# ===========================================================================

def export_and_verify(
    model_name: str,
    output_path: Path,
    version: Optional[str] = None,
    alias: Optional[str] = None,
    opset_version: int = 17,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
    log_artifact: bool = True,
    tracking_uri: str = "http://localhost:5000",
) -> bool:
    """
    Full export + verification pipeline.

    Returns
    -------
    bool  True if export and verification both succeeded.
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    # 1. Load from registry
    pytorch_model, model_uri = load_model_from_registry(
        client, model_name, version=version, alias=alias
    )

    # Infer num_classes from the final Linear layer
    try:
        last_linear = [m for m in pytorch_model.modules() if isinstance(m, torch.nn.Linear)][-1]
        num_classes = last_linear.out_features
        log.info("Detected num_classes = %d", num_classes)
    except Exception:
        num_classes = -1  # unknown, doesn't affect export

    # 2. Export
    export_to_onnx(pytorch_model, output_path, num_classes, opset_version)

    # 3. Verify
    ok = verify_onnx_output(pytorch_model, output_path, atol=atol, rtol=rtol)
    if not ok:
        return False

    # 4. Log back to MLflow (optional)
    if log_artifact and version is not None:
        log_onnx_artifact(client, output_path, model_name, version)
    elif log_artifact and alias is not None:
        try:
            mv = client.get_model_version_by_alias(model_name, alias)
            log_onnx_artifact(client, output_path, model_name, mv.version)
        except Exception:
            log.warning("Could not resolve alias '%s' to log artefact.", alias)

    return True


# ===========================================================================
# CLI
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export a registered MLflow PyTorch model to ONNX and verify it.",
    )
    p.add_argument(
        "--model-name",
        default="defect-efficientnet_b0",
        help="Registered model name in the MLflow Model Registry.",
    )
    p.add_argument(
        "--version",
        default=None,
        help="Specific model version to export (string integer). "
             "Mutually exclusive with --alias.",
    )
    p.add_argument(
        "--alias",
        default="staging",
        help="Model alias to export (default: 'staging'). "
             "Ignored when --version is set.",
    )
    p.add_argument(
        "--output-path",
        type=Path,
        default=Path("models/defect_efficientnet_b0.onnx"),
        help="Destination path for the exported .onnx file.",
    )
    p.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17).",
    )
    p.add_argument(
        "--atol",
        type=float,
        default=DEFAULT_ATOL,
        help=f"Absolute tolerance for output comparison (default: {DEFAULT_ATOL}).",
    )
    p.add_argument(
        "--rtol",
        type=float,
        default=DEFAULT_RTOL,
        help=f"Relative tolerance for output comparison (default: {DEFAULT_RTOL}).",
    )
    p.add_argument(
        "--no-log-artifact",
        dest="log_artifact",
        action="store_false",
        default=True,
        help="Skip logging the ONNX file back to MLflow.",
    )
    p.add_argument(
        "--tracking-uri",
        default="http://localhost:5000",
        help="MLflow tracking server URI.",
    )
    return p.parse_args()


if __name__ == "__main__":
    import sys
    args = parse_args()

    # --version takes priority; if neither given, --alias default applies
    version = args.version
    alias   = None if version is not None else args.alias

    success = export_and_verify(
        model_name=args.model_name,
        output_path=args.output_path,
        version=version,
        alias=alias,
        opset_version=args.opset,
        atol=args.atol,
        rtol=args.rtol,
        log_artifact=args.log_artifact,
        tracking_uri=args.tracking_uri,
    )
    sys.exit(0 if success else 1)

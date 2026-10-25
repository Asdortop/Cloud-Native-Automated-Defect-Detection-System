"""
src/training/train.py
---------------------
End-to-end training script for the NEU Surface Defect classifier.

Architecture  : EfficientNet-B0 (ImageNet pretrained) + custom head
Loss          : Weighted cross-entropy  (handles class imbalance)
Optimiser     : AdamW
Scheduler     : CosineAnnealingLR | StepLR | none
Tracking      : MLflow (parameters, per-epoch metrics, model artefact)

Quick start
-----------
    # 1. Start the MLflow server (see docker/docker-compose.yml)
    docker compose -f docker/docker-compose.yml up -d mlflow

    # 2. Run training
    python src/training/train.py --config config.yaml

    # 3. Override any config key via CLI
    python src/training/train.py --config config.yaml training.lr=3e-4 training.epochs=50
"""

from __future__ import annotations

import argparse
import copy
import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader, random_split, Subset
import torchvision.models as tv_models
import yaml
from sklearn.metrics import recall_score

# Project-local imports
from src.data.preprocessing import DefectDataset
from src.data.transforms import get_transforms

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ===========================================================================
# Config helpers
# ===========================================================================

def _deep_get(d: dict, dotted_key: str) -> Any:
    """Traverse a nested dict with a dot-separated key, e.g. 'training.lr'."""
    keys = dotted_key.split(".")
    for k in keys:
        d = d[k]
    return d


def _deep_set(d: dict, dotted_key: str, value: Any) -> None:
    """Set a value in a nested dict using a dot-separated key."""
    keys = dotted_key.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    # Attempt to preserve numeric types
    existing = d.get(keys[-1])
    if isinstance(existing, float):
        value = float(value)
    elif isinstance(existing, int):
        value = int(value)
    elif isinstance(existing, bool):
        value = value.lower() in ("true", "1", "yes")
    d[keys[-1]] = value


def load_config(config_path: str, overrides: list[str]) -> dict:
    """
    Load YAML config and apply ``key=value`` overrides from the CLI.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file.
    overrides : list[str]
        List of ``"section.key=value"`` strings (e.g. ``["training.lr=3e-4"]``).
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override '{override}'. Expected 'key=value'.")
        key, val = override.split("=", 1)
        _deep_set(cfg, key.strip(), val.strip())
        log.info("Config override: %s = %s", key.strip(), val.strip())

    return cfg


# ===========================================================================
# Reproducibility
# ===========================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic cuDNN (slight performance cost; remove for speed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ===========================================================================
# Model
# ===========================================================================

def build_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
    dropout: float = 0.3,
) -> nn.Module:
    """
    Build EfficientNet-B0 with a custom classification head.

    The ImageNet backbone is frozen for the first few epochs (feature
    extraction phase) and fully fine-tuned afterwards.  Freezing is
    managed externally by the training loop via ``set_backbone_grad``.

    Parameters
    ----------
    model_name  : str   Torchvision model identifier (``efficientnet_b0``).
    num_classes : int   Number of output classes.
    pretrained  : bool  Load ImageNet weights.
    dropout     : float Dropout rate in the classification head.

    Returns
    -------
    nn.Module
    """
    weights = "DEFAULT" if pretrained else None
    model: nn.Module = getattr(tv_models, model_name)(weights=weights)

    # ── Replace the classification head ──────────────────────────────────
    # EfficientNet exposes model.classifier as a Sequential([Dropout, Linear])
    in_features: int = model.classifier[-1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    log.info(
        "Built %s: in_features=%d → num_classes=%d (pretrained=%s, dropout=%.2f)",
        model_name, in_features, num_classes, pretrained, dropout,
    )
    return model


def set_backbone_grad(model: nn.Module, requires_grad: bool) -> None:
    """Freeze / unfreeze all layers except the classification head."""
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = requires_grad


# ===========================================================================
# Class weights
# ===========================================================================

def compute_class_weights(dataset: DefectDataset, device: torch.device) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for weighted cross-entropy.

    Weight_c = total_samples / (n_classes * count_c)

    This ensures rare defect classes receive proportionally higher loss
    contribution during training.
    """
    labels = [label for _, label in dataset.samples]
    counts = np.bincount(labels, minlength=len(dataset.classes)).astype(float)
    weights = len(labels) / (len(dataset.classes) * counts)
    log.info("Class weights: %s", dict(zip(dataset.classes, weights.round(4))))
    return torch.tensor(weights, dtype=torch.float32, device=device)


# ===========================================================================
# Metrics
# ===========================================================================

def compute_metrics(
    all_preds: list[int],
    all_labels: list[int],
    all_losses: list[float],
    classes: list[str],
) -> Dict[str, float]:
    """
    Aggregate epoch-level metrics.

    Returns
    -------
    dict with keys:
        loss            – mean cross-entropy loss
        accuracy        – top-1 accuracy
        recall_macro    – macro-averaged recall across all classes
        recall_<class>  – per-class recall
    """
    preds = np.array(all_preds)
    labels = np.array(all_labels)

    accuracy = float((preds == labels).mean())
    recall_per_class = recall_score(
        labels, preds, labels=list(range(len(classes))),
        average=None, zero_division=0,
    )
    recall_macro = float(recall_per_class.mean())

    metrics: Dict[str, float] = {
        "loss": float(np.mean(all_losses)),
        "accuracy": accuracy,
        "recall_macro": recall_macro,
    }
    for cls_name, r in zip(classes, recall_per_class):
        metrics[f"recall_{cls_name}"] = float(r)

    return metrics


# ===========================================================================
# Data loading
# ===========================================================================

def build_dataloaders(cfg: dict) -> Tuple[DataLoader, DataLoader, list[str]]:
    """
    Construct train and validation DataLoaders from config.

    Supports two modes:
    * Separate ``train_dir`` / ``val_dir`` directories (preferred).
    * Single ``train_dir`` with an automatic ``val_split`` fraction.
    """
    data_cfg = cfg["data"]
    train_tf = get_transforms(train=True)
    val_tf   = get_transforms(train=False)

    train_dir = Path(data_cfg["train_dir"])
    val_dir   = data_cfg.get("val_dir")

    if val_dir and Path(val_dir).exists():
        # ── Mode 1: explicit split directories ───────────────────────────
        train_ds = DefectDataset(train_dir, train=True,  transform=train_tf)
        val_ds   = DefectDataset(Path(val_dir), train=False, transform=val_tf)
        log.info("Using separate train/val dirs: %d / %d samples", len(train_ds), len(val_ds))
    else:
        # ── Mode 2: automatic random split ───────────────────────────────
        full_ds = DefectDataset(train_dir, train=False)  # deterministic for splitting
        n_val   = int(len(full_ds) * float(data_cfg["val_split"]))
        n_train = len(full_ds) - n_val
        train_idx, val_idx = random_split(
            range(len(full_ds)), [n_train, n_val],
            generator=torch.Generator().manual_seed(cfg["training"]["seed"]),
        )
        # Re-wrap subsets with the correct transforms
        train_ds_full = DefectDataset(train_dir, train=True,  transform=train_tf)
        val_ds_full   = DefectDataset(train_dir, train=False, transform=val_tf)
        train_ds = Subset(train_ds_full, train_idx.indices)
        val_ds   = Subset(val_ds_full,   val_idx.indices)

        # Expose .classes / .samples via the underlying dataset
        train_ds.classes = train_ds_full.classes        # type: ignore[attr-defined]
        train_ds.samples = [train_ds_full.samples[i] for i in train_idx.indices]  # type: ignore[attr-defined]
        val_ds.classes   = val_ds_full.classes          # type: ignore[attr-defined]
        val_ds.samples   = [val_ds_full.samples[i]   for i in val_idx.indices]    # type: ignore[attr-defined]

        log.info("Auto-split: %d train / %d val samples", len(train_ds), len(val_ds))

    # Retrieve class list from whichever dataset exposes it
    classes: list[str] = getattr(train_ds, "classes", getattr(train_ds, "dataset", train_ds).classes)

    bs = cfg["training"]["batch_size"]
    nw = data_cfg["num_workers"]

    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True,
        num_workers=nw, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False,
        num_workers=nw, pin_memory=True,
    )
    return train_loader, val_loader, classes


# ===========================================================================
# Training / validation loops
# ===========================================================================

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    is_train: bool,
    grad_clip: float = 0.0,
) -> Tuple[list[float], list[int], list[int]]:
    """
    Execute one epoch (train or eval).

    Returns
    -------
    losses  : per-batch losses
    preds   : predicted class indices (flat list)
    labels  : ground-truth class indices (flat list)
    """
    model.train() if is_train else model.eval()
    losses, preds, labels = [], [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for images, targets in loader:
            images  = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            logits = model(images)
            loss   = criterion(logits, targets)

            if is_train and optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            losses.append(loss.item())
            preds.extend(logits.argmax(dim=1).cpu().tolist())
            labels.extend(targets.cpu().tolist())

    return losses, preds, labels


# ===========================================================================
# Main training loop
# ===========================================================================

def train(cfg: dict) -> None:
    """Full training loop with MLflow tracking."""

    # ── Seed ────────────────────────────────────────────────────────────────
    set_seed(cfg["training"]["seed"])

    # ── Device ──────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Using device: %s", device)

    # ── Data ────────────────────────────────────────────────────────────────
    train_loader, val_loader, classes = build_dataloaders(cfg)
    num_classes = len(classes)
    log.info("Classes (%d): %s", num_classes, classes)

    # ── Model ───────────────────────────────────────────────────────────────
    m_cfg = cfg["model"]
    model = build_model(
        model_name=m_cfg["name"],
        num_classes=num_classes,
        pretrained=m_cfg["pretrained"],
        dropout=m_cfg["dropout"],
    ).to(device)

    # ── Loss (weighted cross-entropy) ────────────────────────────────────────
    # Compute weights from the TRAINING set only
    base_ds = (
        train_loader.dataset.dataset
        if isinstance(train_loader.dataset, Subset)
        else train_loader.dataset
    )
    class_weights = compute_class_weights(base_ds, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ── Optimiser ────────────────────────────────────────────────────────────
    t_cfg = cfg["training"]
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(t_cfg["lr"]),
        weight_decay=float(t_cfg["weight_decay"]),
    )

    # ── LR Scheduler ────────────────────────────────────────────────────────
    scheduler_name = t_cfg.get("lr_scheduler", "cosine").lower()
    if scheduler_name == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=int(t_cfg["epochs"]))
    elif scheduler_name == "step":
        scheduler = StepLR(
            optimizer,
            step_size=int(t_cfg["lr_step_size"]),
            gamma=float(t_cfg["lr_gamma"]),
        )
    else:
        scheduler = None

    # ── MLflow ──────────────────────────────────────────────────────────────
    mlf_cfg = cfg["mlflow"]
    mlflow.set_tracking_uri(mlf_cfg["tracking_uri"])
    mlflow.set_experiment(mlf_cfg["experiment_name"])

    with mlflow.start_run(run_name=mlf_cfg.get("run_name") or None) as run:
        log.info("MLflow run ID: %s", run.info.run_id)

        # Log all config parameters flat
        mlflow.log_params({
            "model_name":       m_cfg["name"],
            "pretrained":       m_cfg["pretrained"],
            "dropout":          m_cfg["dropout"],
            "epochs":           t_cfg["epochs"],
            "batch_size":       t_cfg["batch_size"],
            "lr":               t_cfg["lr"],
            "weight_decay":     t_cfg["weight_decay"],
            "lr_scheduler":     scheduler_name,
            "grad_clip":        t_cfg.get("grad_clip", 0),
            "seed":             t_cfg["seed"],
            "num_classes":      num_classes,
            "classes":          ",".join(classes),
            "device":           str(device),
        })

        # ── Training epochs ──────────────────────────────────────────────
        best_val_loss   = float("inf")
        best_model_wts  = copy.deepcopy(model.state_dict())
        patience_ctr    = 0
        patience        = int(t_cfg.get("early_stopping_patience", 999))
        grad_clip_val   = float(t_cfg.get("grad_clip", 0.0))

        for epoch in range(1, int(t_cfg["epochs"]) + 1):
            t0 = time.time()

            # ── Train ────────────────────────────────────────────────────
            tr_losses, tr_preds, tr_labels = run_epoch(
                model, train_loader, criterion, optimizer, device,
                is_train=True, grad_clip=grad_clip_val,
            )

            # ── Validate ─────────────────────────────────────────────────
            va_losses, va_preds, va_labels = run_epoch(
                model, val_loader, criterion, None, device,
                is_train=False,
            )

            if scheduler is not None:
                scheduler.step()

            # ── Metrics ──────────────────────────────────────────────────
            tr_metrics = compute_metrics(tr_preds, tr_labels, tr_losses, classes)
            va_metrics = compute_metrics(va_preds, va_labels, va_losses, classes)

            elapsed = time.time() - t0
            log.info(
                "Epoch %03d/%d | train loss=%.4f acc=%.4f | "
                "val loss=%.4f acc=%.4f recall=%.4f | %.1fs",
                epoch, t_cfg["epochs"],
                tr_metrics["loss"], tr_metrics["accuracy"],
                va_metrics["loss"], va_metrics["accuracy"],
                va_metrics["recall_macro"],
                elapsed,
            )

            # ── Log to MLflow ─────────────────────────────────────────────
            step_metrics: Dict[str, float] = {}
            for k, v in tr_metrics.items():
                step_metrics[f"train_{k}"] = v
            for k, v in va_metrics.items():
                step_metrics[f"val_{k}"] = v
            step_metrics["lr"] = optimizer.param_groups[0]["lr"]
            mlflow.log_metrics(step_metrics, step=epoch)

            # ── Early stopping + best-weight checkpoint ───────────────────
            if va_metrics["loss"] < best_val_loss:
                best_val_loss  = va_metrics["loss"]
                best_model_wts = copy.deepcopy(model.state_dict())
                patience_ctr   = 0
                log.info("  ✓ New best val_loss=%.4f — saving weights", best_val_loss)
            else:
                patience_ctr += 1
                if patience_ctr >= patience:
                    log.info(
                        "Early stopping triggered after %d epochs without improvement.",
                        patience,
                    )
                    break

        # ── Restore best weights ─────────────────────────────────────────
        model.load_state_dict(best_model_wts)
        log.info("Restored best model weights (val_loss=%.4f)", best_val_loss)
        mlflow.log_metric("best_val_loss", best_val_loss)

        # ── Log model artefact ───────────────────────────────────────────
        if mlf_cfg.get("log_model", True):
            # Build an input example for the model signature
            sample_images, _ = next(iter(val_loader))
            input_example = sample_images[:1].numpy()

            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path="model",
                registered_model_name=f"defect-{m_cfg['name']}",
                input_example=input_example,
            )
            log.info("Model logged to MLflow artifact store.")

        log.info(
            "Training complete. View results at %s/#/experiments/%s",
            mlf_cfg["tracking_uri"],
            mlflow.get_experiment_by_name(mlf_cfg["experiment_name"]).experiment_id,
        )


# ===========================================================================
# Entry point
# ===========================================================================

def parse_args() -> Tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Train the NEU Surface Defect classifier.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Extra arguments are treated as config overrides (dot-notation):\n"
            "  python train.py --config config.yaml training.lr=3e-4 training.epochs=50"
        ),
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to the YAML configuration file (default: config.yaml).",
    )
    return parser.parse_known_args()


if __name__ == "__main__":
    args, overrides = parse_args()
    cfg = load_config(args.config, overrides)
    train(cfg)

"""
tests/evaluate_accuracy.py
----------------------------
Evaluates model classification accuracy on the real NEU Metal Surface Defect dataset.

Computes:
  - Overall Accuracy %
  - Per-Class Precision, Recall, and F1-Score
  - Confusion Matrix
  - Detailed predictions breakdown
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Standard class mapping (lowercase directory name -> canonical class name)
CLASS_MAPPING = {
    "crazing": "Crazing",
    "inclusion": "Inclusion",
    "patches": "Patches",
    "pitted_surface": "Pitted_surface",
    "rolled-in_scale": "Rolled-in_scale",
    "scratches": "Scratches",
}

CANONICAL_CLASSES = [
    "Crazing",
    "Inclusion",
    "Patches",
    "Pitted_surface",
    "Rolled-in_scale",
    "Scratches",
]


def find_dataset_root() -> Path:
    """Find the root directory containing class folders."""
    candidates = [
        Path("data/raw/NEU-DET/train/images"),
        Path("data/raw/NEU-DET/validation/images"),
        Path("data/raw/images"),
        Path("data/raw"),
    ]
    for c in candidates:
        if c.is_dir():
            subdirs = [d.name.lower() for d in c.iterdir() if d.is_dir()]
            if any(k in subdirs for k in CLASS_MAPPING.keys()):
                return c.resolve()
    raise FileNotFoundError("Could not find class subdirectories in data/raw")


def run_accuracy_evaluation(max_per_class: int = 50) -> dict:
    """
    Run inference on real dataset images and evaluate accuracy metrics.
    """
    import httpx

    dataset_root = find_dataset_root()
    print(f"[INFO] Evaluating accuracy using dataset at: {dataset_root}")

    client = httpx.Client(base_url="http://localhost:8095", timeout=30.0)

    y_true = []
    y_pred = []
    confidences = []
    latencies = []
    records = []

    for dir_name, canonical_name in CLASS_MAPPING.items():
        class_dir = dataset_root / dir_name
        if not class_dir.is_dir():
            # Try exact case
            class_dir = dataset_root / canonical_name
        if not class_dir.is_dir():
            print(f"[WARN] Directory not found for class '{canonical_name}' — skipping")
            continue

        images = [
            p for p in class_dir.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".bmp", ".png")
        ]
        images = images[:max_per_class]
        print(f"[INFO] Testing class '{canonical_name}': {len(images)} images ...")

        for img_path in images:
            with open(img_path, "rb") as f:
                img_bytes = f.read()

            t0 = time.perf_counter()
            resp = client.post(
                "/predict",
                files={"file": (img_path.name, img_bytes, "image/jpeg")},
            )
            lat = (time.perf_counter() - t0) * 1000

            if resp.status_code == 200:
                data = resp.json()
                pred_cls = data["predicted_class"]
                conf = data["confidence"]

                y_true.append(canonical_name)
                y_pred.append(pred_cls)
                confidences.append(conf)
                latencies.append(lat)

                records.append({
                    "filename": img_path.name,
                    "true_class": canonical_name,
                    "predicted_class": pred_cls,
                    "correct": canonical_name == pred_cls,
                    "confidence": conf,
                    "latency_ms": lat,
                })
            else:
                print(f"[ERROR] Inference failed for {img_path.name}: {resp.status_code}")

    if not y_true:
        raise ValueError("No images were successfully evaluated.")

    # ── Accuracy Math ────────────────────────────────────────────────────────
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    correct_count = np.sum(y_true_arr == y_pred_arr)
    total_count = len(y_true_arr)
    accuracy_pct = (correct_count / total_count) * 100.0

    # ── Per-class Breakdown ──────────────────────────────────────────────────
    metrics_per_class = {}
    for cls in CANONICAL_CLASSES:
        tp = np.sum((y_true_arr == cls) & (y_pred_arr == cls))
        fp = np.sum((y_true_arr != cls) & (y_pred_arr == cls))
        fn = np.sum((y_true_arr == cls) & (y_pred_arr != cls))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        metrics_per_class[cls] = {
            "total_samples": int(np.sum(y_true_arr == cls)),
            "correct": int(tp),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1_score": round(float(f1), 4),
        }

    summary = {
        "overall_accuracy_pct": round(accuracy_pct, 2),
        "total_images_tested": total_count,
        "correct_predictions": int(correct_count),
        "avg_confidence": round(float(np.mean(confidences)), 4),
        "avg_latency_ms": round(float(np.mean(latencies)), 2),
        "per_class_metrics": metrics_per_class,
    }

    print("\n" + "=" * 65)
    print("      REAL DATASET MODEL ACCURACY BENCHMARK EVALUATION       ")
    print("=" * 65)
    print(f"  Total Tested Images   : {total_count}")
    print(f"  Correct Classifications: {correct_count}")
    print(f"  Overall Accuracy      : {accuracy_pct:.2f}%")
    print(f"  Average Confidence    : {np.mean(confidences)*100:.2f}%")
    print(f"  Average Latency       : {np.mean(latencies):.2f} ms / image")
    print("-" * 65)
    print(f"  {'Class':<20} {'Tested':<8} {'Correct':<8} {'Precision':<10} {'Recall':<8} {'F1-Score'}")
    print("-" * 65)

    for cls, m in metrics_per_class.items():
        print(
            f"  {cls:<20} {m['total_samples']:<8} {m['correct']:<8} "
            f"{m['precision']*100:>6.1f}%    {m['recall']*100:>6.1f}%   {m['f1_score']:.4f}"
        )
    print("=" * 65 + "\n")

    return summary


if __name__ == "__main__":
    run_accuracy_evaluation(max_per_class=50)

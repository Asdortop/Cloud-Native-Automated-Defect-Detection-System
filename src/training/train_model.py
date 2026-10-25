"""
src/training/train_model.py
---------------------------
Train EfficientNet-B0 on the NEU Metal Surface Defect dataset and export the
trained model weights to models/defect_efficientnet_b0.onnx.
"""

import copy
import logging
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split, Subset
import torchvision.models as tv_models

from src.data.preprocessing import DefectDataset
from src.data.transforms import get_transforms

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def find_dataset_dir() -> Path:
    candidates = [
        Path("data/raw/NEU-DET/train/images"),
        Path("data/raw/NEU-DET/validation/images"),
        Path("data/raw/images"),
        Path("data/raw"),
    ]
    for c in candidates:
        if c.is_dir():
            subdirs = [d.name.lower() for d in c.iterdir() if d.is_dir()]
            if any(k in subdirs for k in ["crazing", "inclusion", "patches", "pitted_surface"]):
                return c.resolve()
    raise FileNotFoundError("Could not find defect class subdirectories in data/raw")


def train_and_export(epochs: int = 5, batch_size: int = 32, lr: float = 1e-3):
    dataset_root = find_dataset_dir()
    log.info("Loading training dataset from: %s", dataset_root)

    train_tf = get_transforms(train=True)
    val_tf   = get_transforms(train=False)

    full_ds = DefectDataset(dataset_root, train=False)
    classes = full_ds.classes
    num_classes = len(classes)
    log.info("Classes (%d): %s", num_classes, classes)

    # 80/20 train/val split
    n_val = int(len(full_ds) * 0.2)
    n_train = len(full_ds) - n_val
    train_idx, val_idx = random_split(
        range(len(full_ds)), [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    train_ds_full = DefectDataset(dataset_root, train=True, transform=train_tf)
    val_ds_full   = DefectDataset(dataset_root, train=False, transform=val_tf)

    train_ds = Subset(train_ds_full, train_idx.indices)
    val_ds   = Subset(val_ds_full, val_idx.indices)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Training on device: %s", device)

    # Load ImageNet Pretrained EfficientNet-B0
    model = tv_models.efficientnet_b0(weights=tv_models.EfficientNet_B0_Weights.DEFAULT)
    in_features = model.classifier[-1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    model = model.to(device)

    # Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0

        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            tr_loss += loss.item() * len(targets)
            tr_correct += (outputs.argmax(dim=1) == targets).sum().item()
            tr_total += len(targets)

        scheduler.step()

        # Validation
        model.eval()
        va_loss, va_correct, va_total = 0.0, 0, 0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)

                va_loss += loss.item() * len(targets)
                va_correct += (outputs.argmax(dim=1) == targets).sum().item()
                va_total += len(targets)

        tr_acc = (tr_correct / tr_total) * 100.0
        va_acc = (va_correct / va_total) * 100.0
        elapsed = time.time() - t0

        log.info(
            "Epoch %d/%d | Train Acc: %.2f%% | Val Acc: %.2f%% | Val Loss: %.4f | Time: %.1fs",
            epoch, epochs, tr_acc, va_acc, va_loss / va_total, elapsed,
        )

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_weights = copy.deepcopy(model.state_dict())

    # Export best model to ONNX
    model.load_state_dict(best_weights)
    model.eval().cpu()

    # Save PyTorch checkpoint and export clean ONNX file
    torch.save(model.state_dict(), "models/defect_efficientnet_b0.pt")
    
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=14,
        dynamo=False,
    )

    log.info("Successfully trained model (Best Val Acc: %.2f%%) and exported ONNX model to %s", best_val_acc, onnx_path)


if __name__ == "__main__":
    train_and_export(epochs=5, batch_size=32, lr=1e-3)

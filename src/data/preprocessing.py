"""
preprocessing.py
----------------
PyTorch Dataset for the NEU Metal Surface Defect dataset.

Expected directory layout
--------------------------
    root/
        Crazing/
            img_001.jpg
            img_002.jpg
            ...
        Inclusion/
            ...
        Patches/
            ...
        Pitted_surface/
            ...
        Rolled-in_scale/
            ...
        Scratches/
            ...

Each sub-directory name becomes the class label.  Any folder found under
*root* is treated as a class (no hard-coded class list).

Usage
-----
    from src.data.preprocessing import DefectDataset

    train_ds = DefectDataset("data/raw", train=True)
    val_ds   = DefectDataset("data/raw", train=False)

    loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset

from src.data.transforms import get_transforms

# Supported image extensions (case-insensitive)
_IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _is_image(path: Path) -> bool:
    """Return ``True`` if *path* has a recognised image extension."""
    return path.suffix.lower() in _IMG_EXTENSIONS


class DefectDataset(Dataset):
    """
    Dataset for image-based surface defect classification.

    The dataset scans *root* for sub-directories; each sub-directory is
    treated as a distinct class.  Class indices are assigned in sorted
    alphabetical order so they remain stable across runs.

    Parameters
    ----------
    root : str | Path
        Path to the dataset root directory containing one sub-folder per
        defect class.
    train : bool, optional
        When ``True`` (default) the pipeline includes stochastic
        augmentations (random flip, rotation, colour jitter).  Set to
        ``False`` for validation / inference to get a deterministic output.
    transform : Callable | None, optional
        Override the default transform pipeline.  If ``None`` (default)
        ``get_transforms(train)`` is used.  Pass a custom
        ``torchvision.transforms.Compose`` to replace it entirely.

    Attributes
    ----------
    classes : List[str]
        Sorted list of class names (sub-directory names).
    class_to_idx : Dict[str, int]
        Mapping from class name → integer index.
    samples : List[Tuple[Path, int]]
        List of ``(image_path, class_index)`` pairs for the full dataset.
    """

    def __init__(
        self,
        root: str | Path,
        train: bool = True,
        transform: Optional[Callable] = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.train = train
        self.transform: Callable = transform if transform is not None else get_transforms(train)

        if not self.root.exists():
            raise FileNotFoundError(
                f"Dataset root not found: '{self.root}'. "
                "Run src/data/download_dataset.py first."
            )

        self.classes, self.class_to_idx = self._discover_classes()
        self.samples: List[Tuple[Path, int]] = self._build_sample_list()

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No images found under '{self.root}'. "
                f"Searched extensions: {sorted(_IMG_EXTENSIONS)}"
            )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _discover_classes(self) -> Tuple[List[str], Dict[str, int]]:
        """Scan *root* for class sub-directories (sorted, stable order)."""
        classes = sorted(
            entry.name
            for entry in self.root.iterdir()
            if entry.is_dir() and not entry.name.startswith(".")
        )
        if not classes:
            raise RuntimeError(
                f"No class sub-directories found under '{self.root}'."
            )
        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        return classes, class_to_idx

    def _build_sample_list(self) -> List[Tuple[Path, int]]:
        """Walk each class directory and collect (image_path, label) pairs."""
        samples: List[Tuple[Path, int]] = []
        for cls_name, cls_idx in self.class_to_idx.items():
            cls_dir = self.root / cls_name
            for img_path in sorted(cls_dir.iterdir()):   # sorted → reproducible order
                if _is_image(img_path):
                    samples.append((img_path, cls_idx))
        return samples

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        Parameters
        ----------
        index : int
            Sample index.

        Returns
        -------
        image : torch.Tensor
            Normalised image tensor of shape ``(3, 224, 224)``.
        label : int
            Integer class index.
        """
        img_path, label = self.samples[index]

        # Load as RGB to handle greyscale / RGBA inputs uniformly
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        return image, label

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"root='{self.root}', "
            f"train={self.train}, "
            f"classes={self.classes}, "
            f"n_samples={len(self.samples)})"
        )

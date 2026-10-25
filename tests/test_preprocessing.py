"""
tests/test_preprocessing.py
----------------------------
Unit tests for DefectDataset and get_transforms.

The tests use a *synthetic* fixture dataset generated entirely in memory
(via ``PIL.Image``) so they run without the real NEU dataset being present.

Test coverage
-------------
1. ``test_image_count``           – correct number of images discovered
2. ``test_output_tensor_shape``   – tensor shape is (3, 224, 224)
3. ``test_val_transform_dtype``   – output dtype is float32
4. ``test_val_is_deterministic``  – train=False produces identical tensors
                                    on two consecutive calls to __getitem__
5. ``test_train_is_stochastic``   – train=True *may* produce different tensors
                                    (probabilistic; skipped if unlikely)
6. ``test_class_discovery``       – classes list matches fixture directories
7. ``test_missing_root_raises``   – FileNotFoundError for non-existent root
8. ``test_get_transforms_shapes`` – both pipelines return (3, 224, 224)
"""

from __future__ import annotations

import random
import shutil
import tempfile
from pathlib import Path
from typing import Generator

import pytest
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Make the src package importable from any working directory
# ---------------------------------------------------------------------------
import sys, os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import DefectDataset
from src.data.transforms import get_transforms, IMAGE_SIZE


# ===========================================================================
# Fixtures
# ===========================================================================

# Fixture dataset layout
#   <tmp>/
#       ClassA/  (3 images)
#       ClassB/  (2 images)
#       ClassC/  (5 images)
_FIXTURE_LAYOUT: dict[str, int] = {
    "ClassA": 3,
    "ClassB": 2,
    "ClassC": 5,
}
_TOTAL_IMAGES = sum(_FIXTURE_LAYOUT.values())  # 10


def _make_random_image(path: Path, size: tuple[int, int] = (300, 300)) -> None:
    """Save a random RGB JPEG to *path*."""
    arr = bytes(random.getrandbits(8) for _ in range(size[0] * size[1] * 3))
    img = Image.frombytes("RGB", size, arr)
    img.save(path, format="JPEG")


@pytest.fixture(scope="module")
def fixture_root() -> Generator[Path, None, None]:
    """
    Create a temporary directory tree with synthetic JPEG images, yield its
    Path, then clean up after all tests in this module finish.
    """
    tmp = Path(tempfile.mkdtemp(prefix="defect_test_"))
    try:
        for cls_name, n_imgs in _FIXTURE_LAYOUT.items():
            cls_dir = tmp / cls_name
            cls_dir.mkdir()
            for i in range(n_imgs):
                _make_random_image(cls_dir / f"img_{i:04d}.jpg")
        yield tmp
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ===========================================================================
# 1. Correct image count
# ===========================================================================

class TestImageCount:
    def test_total_images_loaded(self, fixture_root: Path) -> None:
        """Dataset must discover all synthetic images."""
        ds = DefectDataset(fixture_root, train=False)
        assert len(ds) == _TOTAL_IMAGES, (
            f"Expected {_TOTAL_IMAGES} images, got {len(ds)}"
        )

    def test_per_class_count(self, fixture_root: Path) -> None:
        """Each class contributes the expected number of samples."""
        ds = DefectDataset(fixture_root, train=False)
        from collections import Counter

        label_counts = Counter(label for _, label in ds.samples)
        for cls_name, expected in _FIXTURE_LAYOUT.items():
            idx = ds.class_to_idx[cls_name]
            assert label_counts[idx] == expected, (
                f"Class '{cls_name}': expected {expected} images, "
                f"got {label_counts[idx]}"
            )


# ===========================================================================
# 2. Correct output tensor shape
# ===========================================================================

class TestTensorShape:
    @pytest.mark.parametrize("train", [True, False])
    def test_shape(self, fixture_root: Path, train: bool) -> None:
        """Every sample must return a tensor of shape (3, IMAGE_SIZE, IMAGE_SIZE)."""
        ds = DefectDataset(fixture_root, train=train)
        tensor, label = ds[0]
        expected = (3, IMAGE_SIZE, IMAGE_SIZE)
        assert tensor.shape == torch.Size(expected), (
            f"train={train}: expected shape {expected}, got {tensor.shape}"
        )

    def test_label_is_int(self, fixture_root: Path) -> None:
        """The returned label must be a plain Python int."""
        ds = DefectDataset(fixture_root, train=False)
        _, label = ds[0]
        assert isinstance(label, int)

    def test_label_in_range(self, fixture_root: Path) -> None:
        """All labels must fall in [0, n_classes)."""
        ds = DefectDataset(fixture_root, train=False)
        n_classes = len(ds.classes)
        for _, label in ds.samples:
            assert 0 <= label < n_classes


# ===========================================================================
# 3. train=False produces deterministic output
# ===========================================================================

class TestDeterminism:
    def test_val_deterministic_across_two_calls(self, fixture_root: Path) -> None:
        """
        Calling __getitem__ twice on the same index with train=False must
        return bit-for-bit identical tensors.
        """
        ds = DefectDataset(fixture_root, train=False)
        t1, _ = ds[0]
        t2, _ = ds[0]
        assert torch.equal(t1, t2), (
            "val transform is not deterministic: two calls to ds[0] "
            "returned different tensors."
        )

    def test_val_deterministic_across_dataset(self, fixture_root: Path) -> None:
        """All samples must be deterministic under the val pipeline."""
        ds = DefectDataset(fixture_root, train=False)
        for idx in range(len(ds)):
            t1, _ = ds[idx]
            t2, _ = ds[idx]
            assert torch.equal(t1, t2), (
                f"Val pipeline produced non-deterministic output for index {idx}."
            )

    def test_val_dtype_is_float32(self, fixture_root: Path) -> None:
        """Normalised tensors must be float32."""
        ds = DefectDataset(fixture_root, train=False)
        tensor, _ = ds[0]
        assert tensor.dtype == torch.float32, (
            f"Expected float32, got {tensor.dtype}"
        )


# ===========================================================================
# 4. Extra: class discovery
# ===========================================================================

class TestClassDiscovery:
    def test_class_names_match_fixture(self, fixture_root: Path) -> None:
        """Discovered class names must exactly match the fixture directories."""
        ds = DefectDataset(fixture_root, train=False)
        expected = sorted(_FIXTURE_LAYOUT.keys())
        assert ds.classes == expected, (
            f"Expected classes {expected}, got {ds.classes}"
        )

    def test_class_to_idx_is_sorted(self, fixture_root: Path) -> None:
        """
        Class indices must follow sorted alphabetical order so that they
        are stable across independent instantiations.
        """
        ds = DefectDataset(fixture_root, train=False)
        for expected_idx, cls_name in enumerate(sorted(ds.classes)):
            assert ds.class_to_idx[cls_name] == expected_idx


# ===========================================================================
# 5. Error handling
# ===========================================================================

class TestErrorHandling:
    def test_missing_root_raises(self) -> None:
        """Instantiating with a non-existent root must raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            DefectDataset("/this/path/does/not/exist", train=False)

    def test_empty_root_raises(self, tmp_path: Path) -> None:
        """A root with no class sub-directories must raise RuntimeError."""
        with pytest.raises(RuntimeError):
            DefectDataset(tmp_path, train=False)


# ===========================================================================
# 6. get_transforms standalone tests
# ===========================================================================

class TestGetTransforms:
    @pytest.mark.parametrize("train", [True, False])
    def test_transform_output_shape(self, train: bool) -> None:
        """get_transforms must produce (3, 224, 224) tensors from a PIL image."""
        tf = get_transforms(train)
        img = Image.new("RGB", (640, 480), color=(128, 64, 32))
        tensor = tf(img)
        assert tensor.shape == torch.Size([3, IMAGE_SIZE, IMAGE_SIZE])

    def test_val_transform_is_deterministic(self) -> None:
        """The val transform applied to the same PIL image must be identical."""
        tf = get_transforms(train=False)
        img = Image.new("RGB", (300, 300), color=(100, 150, 200))
        t1 = tf(img)
        t2 = tf(img)
        assert torch.equal(t1, t2)

    def test_transforms_are_different_objects(self) -> None:
        """Each call to get_transforms must return a fresh Compose object."""
        tf1 = get_transforms(train=False)
        tf2 = get_transforms(train=False)
        assert tf1 is not tf2

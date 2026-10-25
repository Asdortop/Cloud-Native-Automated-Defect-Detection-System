"""
transforms.py
-------------
Centralised transform pipeline for the NEU Surface Defect project.

Importing this module guarantees that the training script and the serving
API always apply **identical** preprocessing.  Only the training pipeline
receives the additional stochastic augmentations.

Usage
-----
    from src.data.transforms import get_transforms

    train_tf = get_transforms(train=True)
    infer_tf = get_transforms(train=False)
"""

from torchvision import transforms

# ---------------------------------------------------------------------------
# ImageNet statistics (used for pre-trained backbone normalisation)
# ---------------------------------------------------------------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

IMAGE_SIZE = 224  # target spatial resolution


def get_transforms(train: bool) -> transforms.Compose:
    """
    Build and return a ``torchvision.transforms.Compose`` pipeline.

    Parameters
    ----------
    train : bool
        * ``True``  – includes stochastic augmentations (flip, rotation,
          colour jitter) suitable for training.
        * ``False`` – deterministic resize + normalise pipeline suitable
          for validation, testing, and production inference.

    Returns
    -------
    torchvision.transforms.Compose
        A ready-to-use transform that accepts a PIL Image and returns a
        normalised ``torch.Tensor`` of shape ``(3, 224, 224)``.
    """
    # ------------------------------------------------------------------
    # Shared tail: always applied regardless of split
    # ------------------------------------------------------------------
    shared_tail = [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]

    if not train:
        return transforms.Compose(shared_tail)

    # ------------------------------------------------------------------
    # Training head: stochastic augmentations prepended before resize
    # ------------------------------------------------------------------
    aug_head = [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05,
        ),
    ]

    return transforms.Compose(aug_head + shared_tail)

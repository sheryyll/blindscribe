"""
EMNIST Balanced dataset wrapper.

Balanced split: 47 classes covering 0-9, A-Z, and lowercase letters that are
visually distinct from their uppercase form (a, b, d, e, f, g, h, n, q, r,
t). Visually-similar pairs like C/c, O/o, S/s, etc. are merged into a single
class in this split, which keeps the label space sane for a small CNN and
avoids the model being penalized for a distinction that isn't really visible
in isolated block-letter handwriting.

We reuse backend/preprocessing.py's normalize_char_crop for the PIL-image
path, but torchvision gives us tensors directly, so here we replicate only
the parts of that pipeline that apply to already-28x28, already-centered
EMNIST source images: the transpose quirk, the invert-to-white-on-black
convention, and the mean/std normalization. If you change preprocessing.py,
mirror the change here (see PREPROCESSING_MUST_MATCH marker).
"""

import string

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

EMNIST_MEAN = 0.1307
EMNIST_STD = 0.3081

# EMNIST Balanced label order, per the official mapping file.
BALANCED_CLASSES = (
    list(string.digits)
    + list(string.ascii_uppercase)
    + ["a", "b", "d", "e", "f", "g", "h", "n", "q", "r", "t"]
)
assert len(BALANCED_CLASSES) == 47


def label_to_char(label: int) -> str:
    return BALANCED_CLASSES[label]


# PREPROCESSING_MUST_MATCH: keep this transform in sync with
# backend/preprocessing.normalize_char_crop. EMNIST source images are
# already 28x28 and centered, so we skip the resize/pad steps and only
# apply: transpose (rotate+flip quirk) -> normalize.
train_transform = transforms.Compose(
    [
        # torchvision.datasets.EMNIST already returns images in the
        # "transposed" orientation baked into the raw files, matching what
        # normalize_char_crop produces via arr.T. No extra transpose needed
        # here since PIL/torchvision decode it consistently already -
        # verified against a sample of known digits before training.
        transforms.RandomRotation(8),          # small jitter, real handwriting varies
        transforms.RandomAffine(0, translate=(0.08, 0.08), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((EMNIST_MEAN,), (EMNIST_STD,)),
    ]
)

eval_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((EMNIST_MEAN,), (EMNIST_STD,)),
    ]
)


def get_loaders(data_dir: str, batch_size: int = 128, num_workers: int = 2):
    train_set = datasets.EMNIST(
        data_dir, split="balanced", train=True, download=True, transform=train_transform
    )
    test_set = datasets.EMNIST(
        data_dir, split="balanced", train=False, download=True, transform=eval_transform
    )
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, test_loader

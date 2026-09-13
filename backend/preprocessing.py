import numpy as np
from PIL import Image, ImageOps

EMNIST_MEAN = 0.1307
EMNIST_STD = 0.3081
TARGET_SIZE = 28
# Leave a small margin around the character so strokes don't touch the
# edge of the frame, matching how EMNIST characters are typically centered.
PADDING_RATIO = 0.15


def crop_to_square(img: Image.Image) -> Image.Image:
    """Pad a PIL image to a square canvas (centered) before resizing.

    Resizing a non-square crop directly to 28x28 distorts aspect ratio,
    which matters a lot for characters like '1' or 'l' vs 'O'. Padding to
    square first, then resizing, preserves shape.
    """
    w, h = img.size
    side = max(w, h)
    pad_w = (side - w) // 2
    pad_h = (side - h) // 2
    return ImageOps.expand(img, border=(pad_w, pad_h, side - w - pad_w, side - h - pad_h), fill=255)


def normalize_char_crop(img: Image.Image) -> np.ndarray:
    """
    Convert a single-character crop (white background, dark strokes,
    any size, mode 'L') into a (1, 28, 28) float32 array ready for the model.
    """
    if img.mode != "L":
        img = img.convert("L")

    # Add a margin, then pad to square, then resize.
    w, h = img.size
    margin = int(max(w, h) * PADDING_RATIO)
    img = ImageOps.expand(img, border=margin, fill=255)
    img = crop_to_square(img)
    img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)

    arr = np.asarray(img, dtype=np.float32) / 255.0

    # EMNIST convention: white strokes on black background (inverted from
    # our canvas convention of dark strokes on white). Invert to match.
    arr = 1.0 - arr

    # EMNIST images are also stored transposed relative to a "natural"
    # reading orientation (a well-known quirk of the original dataset
    # dump). dataset.py applies the same transpose to training images, so
    # this function must apply it too, or the model will see rotated input
    # at inference time despite non-rotated input at training time.
    arr = arr.T

    arr = (arr - EMNIST_MEAN) / EMNIST_STD
    return arr[np.newaxis, :, :]  # (1, 28, 28)

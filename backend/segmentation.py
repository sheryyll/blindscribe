"""
Segments a canvas image (one handwritten string, dark strokes on light
background) into an ordered list of single-character crops.

APPROACH
--------
1. Binarize the image (threshold).
2. Find connected components (8-connectivity) - each blob is a candidate
   character or character-fragment.
3. Merge fragments that belong to the same character:
   - dotted characters (i, j): a small blob sitting above a taller blob,
     horizontally overlapping -> merge into one character.
   - stray tiny specks (< MIN_BLOB_AREA): dropped as noise, not merged.
4. Sort remaining blobs left-to-right by bounding-box x-center to get
   reading order.
5. Crop each with a small margin and hand off to preprocessing.py.

KNOWN FAILURE MODES (documented deliberately, not hidden)
-----------------------------------------------------------
- Touching/overlapping characters (fast cursive-style writing where letters
  visually connect) will be under-segmented into one blob. This heuristic
  assumes reasonably separated print-style characters, which is a
  reasonable v1 constraint for a canvas-drawing use case - the frontend
  prompts the user to write with gaps between characters.
- A character that is itself naturally split into unconnected strokes
  (rare in block print, more common in certain 'k', 'x' renderings with a
  thin waist) may be over-segmented into two blobs. The dotted-character
  merge only fires for the specific small-blob-above-tall-blob pattern, so
  it won't accidentally merge two full-size blobs.
- No rotation/skew correction - assumes roughly horizontal writing, which
  matches a canvas UI where the string is written left-to-right in one line.
"""

from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy import ndimage

BINARIZE_THRESHOLD = 200      # pixel value below this (0-255, grayscale) counts as "ink"
MIN_BLOB_AREA = 8             # px^2; smaller blobs are treated as noise
CROP_MARGIN_PX = 4


@dataclass
class CharBox:
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self):
        return self.x1 - self.x0

    @property
    def height(self):
        return self.y1 - self.y0

    @property
    def x_center(self):
        return (self.x0 + self.x1) / 2

    def horizontally_overlaps(self, other: "CharBox") -> bool:
        return not (self.x1 < other.x0 or other.x1 < self.x0)

    def merged_with(self, other: "CharBox") -> "CharBox":
        return CharBox(
            x0=min(self.x0, other.x0),
            y0=min(self.y0, other.y0),
            x1=max(self.x1, other.x1),
            y1=max(self.y1, other.y1),
        )


def _binary_mask(img: Image.Image) -> np.ndarray:
    gray = np.asarray(img.convert("L"))
    return gray < BINARIZE_THRESHOLD  # True where there's ink


def _find_blobs(mask: np.ndarray) -> list[CharBox]:
    structure = np.ones((3, 3), dtype=int)  # 8-connectivity
    labeled, num_features = ndimage.label(mask, structure=structure)
    boxes = []
    for i in range(1, num_features + 1):
        ys, xs = np.where(labeled == i)
        area = len(xs)
        if area < MIN_BLOB_AREA:
            continue
        boxes.append(CharBox(x0=int(xs.min()), y0=int(ys.min()), x1=int(xs.max()), y1=int(ys.max())))
    return boxes


def _merge_dotted_characters(boxes: list[CharBox]) -> list[CharBox]:
    """
    Merge a small blob (a dot, or a disconnected fragment like a serif or
    crossbar drawn as a separate mouse stroke) with the taller blob it
    actually belongs to, when they overlap horizontally.

    SELECTION METRIC: overlap is measured as a FRACTION OF THE CANDIDATE'S
    OWN WIDTH (overlap_px / candidate.width), not raw pixel overlap. This
    matters because raw pixel overlap is biased toward wide candidates: a
    fragment sitting between its own narrow parent stroke and a wide
    neighboring letter can accumulate more raw overlap pixels against the
    wide neighbor purely because the neighbor is wide, even when the
    fragment barely clips its edge. Normalizing by the candidate's width
    instead asks "what fraction of this candidate's own extent does the
    fragment cover?" - a narrow true-parent stroke that's mostly or fully
    beneath the fragment scores near 1.0, while a wide neighbor that's only
    clipped at the edge scores low, regardless of its raw pixel count.

    This was found empirically: a disconnected 'J' serif (drawn as its own
    mouse stroke, not touching J's vertical stroke) was incorrectly merged
    into a neighboring 'E' because E's width let it win on raw overlap
    despite the serif's true parent stroke being directly beneath it.

    A minimum overlap fraction (MIN_OVERLAP_FRACTION) is also required, so
    a fragment with only incidental, weak overlap against any candidate is
    left unmerged rather than forced onto the least-bad option.
    """
    if len(boxes) < 2:
        return boxes

    DOT_TO_BODY_HEIGHT_RATIO = 0.55  # dot must be well under half the body's height
    MAX_VERTICAL_GAP_RATIO = 1.0     # gap between dot bottom and body top, relative to body height
    MIN_OVERLAP_FRACTION = 0.5       # fragment must cover at least half of the candidate's width

    used = [False] * len(boxes)
    merged: list[CharBox] = []

    for i, box_i in enumerate(boxes):
        if used[i]:
            continue

        best_match = None
        best_fraction = 0.0
        for j, box_j in enumerate(boxes):
            if i == j or used[j]:
                continue
            if box_j.height == 0 or box_j.width == 0:
                continue
            if box_i.height / box_j.height > DOT_TO_BODY_HEIGHT_RATIO:
                continue
            if box_j.y0 < box_i.y1:
                continue  # body must start below where the dot/fragment ends
            gap = box_j.y0 - box_i.y1
            if gap > box_j.height * MAX_VERTICAL_GAP_RATIO:
                continue  # too far below to plausibly be the same character
            if not box_i.horizontally_overlaps(box_j):
                continue

            overlap = min(box_i.x1, box_j.x1) - max(box_i.x0, box_j.x0)
            overlap_fraction = overlap / box_j.width  # normalized by CANDIDATE's width, not the fragment's

            if overlap_fraction >= MIN_OVERLAP_FRACTION and overlap_fraction > best_fraction:
                best_fraction = overlap_fraction
                best_match = j

        if best_match is not None:
            merged.append(box_i.merged_with(boxes[best_match]))
            used[i] = True
            used[best_match] = True

    for i, box in enumerate(boxes):
        if not used[i]:
            merged.append(box)

    return merged


def segment_characters(img: Image.Image) -> list[Image.Image]:
    """
    Full pipeline: image -> ordered list of single-character PIL crops
    (mode 'L', white background, dark ink - same convention preprocessing.py
    expects).
    """
    mask = _binary_mask(img)
    boxes = _find_blobs(mask)
    boxes = _merge_dotted_characters(boxes)
    boxes.sort(key=lambda b: b.x_center)  # reading order, left to right

    gray = img.convert("L")
    w, h = gray.size
    crops = []
    for box in boxes:
        x0 = max(0, box.x0 - CROP_MARGIN_PX)
        y0 = max(0, box.y0 - CROP_MARGIN_PX)
        x1 = min(w, box.x1 + CROP_MARGIN_PX)
        y1 = min(h, box.y1 + CROP_MARGIN_PX)
        crops.append(gray.crop((x0, y0, x1, y1)))

    return crops


def segment_characters_with_boxes(img: Image.Image) -> tuple[list[Image.Image], list[CharBox]]:
    """Same as segment_characters but also returns bounding boxes, so the
    API can report them to the frontend for visualization/debugging."""
    mask = _binary_mask(img)
    boxes = _find_blobs(mask)
    boxes = _merge_dotted_characters(boxes)
    boxes.sort(key=lambda b: b.x_center)

    gray = img.convert("L")
    w, h = gray.size
    crops = []
    for box in boxes:
        x0 = max(0, box.x0 - CROP_MARGIN_PX)
        y0 = max(0, box.y0 - CROP_MARGIN_PX)
        x1 = min(w, box.x1 + CROP_MARGIN_PX)
        y1 = min(h, box.y1 + CROP_MARGIN_PX)
        crops.append(gray.crop((x0, y0, x1, y1)))

    return crops, boxes

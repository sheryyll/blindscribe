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

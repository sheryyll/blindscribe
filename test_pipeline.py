"""
Quick end-to-end sanity check: renders text to an image (simulating
handwriting), runs it through segmentation + classification, and prints
the result. Useful for verifying the full pipeline works after training,
without needing the browser frontend running.

Run from the repo root:
    python test_pipeline.py "ABC"
    python test_pipeline.py "Hi"

Requires backend/model.onnx and backend/model_classes.json to already
exist (see README: model/train.py then model/export_onnx.py).
"""

import sys
from pathlib import Path

# Make backend/ importable regardless of cwd - this script is the
# "supported" way to run ad-hoc tests without hitting the relative-import
# gotchas that come from mixing `from backend.x import y` (package-style)
# with the flat sibling imports used inside backend/*.py (script-style).
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from segmentation import segment_characters_with_boxes  # noqa: E402
from inference import CharacterClassifier  # noqa: E402


def render_text_image(text: str, font_size: int = 100) -> Image.Image:
    """Renders text with a system font as a stand-in for a canvas drawing.
    Not a substitute for testing with real handwriting, but good for
    verifying the pipeline end-to-end without the browser."""
    width = max(400, 120 * len(text))
    img = Image.new("L", (width, 200), 255)
    draw = ImageDraw.Draw(img)

    font_candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",  # macOS
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
        "arial.ttf",  # Windows (if on PATH)
    ]
    font = None
    for path in font_candidates:
        try:
            font = ImageFont.truetype(path, font_size)
            break
        except OSError:
            continue
    if font is None:
        print("Warning: no system font found, falling back to PIL default "
              "(characters will be tiny and may segment incorrectly).")
        font = ImageFont.load_default()

    draw.text((30, 30), text, font=font, fill=0)
    return img


def main():
    text = sys.argv[1] if len(sys.argv) > 1 else "ABC"
    print(f"Rendering test image for: {text!r}")

    img = render_text_image(text)
    crops, boxes = segment_characters_with_boxes(img)
    print(f"Segmented into {len(crops)} character(s) (input had {len(text)} character(s))")

    if len(crops) != len(text):
        print("  Note: segment count doesn't match input length - check "
              "segmentation.py's known failure modes in its docstring, or "
              "try a font/size that separates characters more clearly.")

    classifier = CharacterClassifier()
    predictions = classifier.predict_string(crops)

    print()
    for i, (box, pred) in enumerate(zip(boxes, predictions)):
        alts = ", ".join(f"{a['char']}:{a['confidence']:.2f}" for a in pred["top_alternatives"])
        print(
            f"  [{i}] predicted='{pred['char']}'  confidence={pred['confidence']:.2f}  "
            f"bbox=({box.x0},{box.y0},{box.x1},{box.y1})  top-3=[{alts}]"
        )

    predicted_string = "".join(p["char"] for p in predictions)
    print()
    print(f"Input:     {text}")
    print(f"Predicted: {predicted_string}")


if __name__ == "__main__":
    main()

import base64
import io
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

try:
    from .inference import get_classifier
    from .segmentation import segment_characters_with_boxes
except ImportError:
    from inference import get_classifier
    from segmentation import segment_characters_with_boxes


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_classifier()
    try:
        from .gradcam import get_explainer
    except ImportError:
        from gradcam import get_explainer
    get_explainer()
    yield


app = FastAPI(title="BlidScribe", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    image_base64: str  # PNG, data-URL prefix optional


class ExplainRequest(BaseModel):
    image_base64: str
    char_index: int  # which segmented character to explain, from a prior /predict call


def _decode_image(image_base64: str) -> Image.Image:
    if "," in image_base64 and image_base64.strip().startswith("data:"):
        image_base64 = image_base64.split(",", 1)[1]
    try:
        raw = base64.b64decode(image_base64)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not decode image: {exc}")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    img = _decode_image(req.image_base64)
    crops, boxes = segment_characters_with_boxes(img)

    if not crops:
        return {
            "predicted_string": "",
            "characters": [],
            "warning": "No strokes detected. Try writing larger or with a thicker line.",
        }

    classifier = get_classifier()
    predictions = classifier.predict_string(crops)

    characters = []
    for pred, box in zip(predictions, boxes):
        characters.append(
            {
                "char": pred["char"],
                "confidence": pred["confidence"],
                "top_alternatives": pred["top_alternatives"],
                "bbox": {"x0": box.x0, "y0": box.y0, "x1": box.x1, "y1": box.y1},
            }
        )

    predicted_string = "".join(c["char"] for c in characters)

    return {
        "predicted_string": predicted_string,
        "characters": characters,
    }


@app.post("/explain")
def explain(req: ExplainRequest):
    try:
        from .gradcam import get_explainer
    except ImportError:
        from gradcam import get_explainer

    img = _decode_image(req.image_base64)
    crops, _ = segment_characters_with_boxes(img)

    if req.char_index < 0 or req.char_index >= len(crops):
        raise HTTPException(status_code=400, detail="char_index out of range")

    explainer = get_explainer()
    result = explainer.explain(crops[req.char_index])
    return result

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

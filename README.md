# BlidScribe

Draw a word or string of characters on a canvas → the backend segments it into
individual characters, classifies each one with a CNN trained on EMNIST, and
reassembles the predicted string — with per-character confidence and
Grad-CAM explainability.

This is an end-to-end project: dataset → trained model → ONNX export →
FastAPI inference service → browser canvas client. Unlike a plain MNIST demo,
the interesting engineering problem here is **segmentation and train/serve
consistency**, not just classification accuracy.

## Architecture

```
Canvas (browser)
   │  draw strokes, export PNG
   ▼
POST /predict  ──────────────► FastAPI (backend/app.py)
                                     │
                                     ▼
                         segmentation.py
                    (binarize → connected components
                     → merge dotted chars like i/j
                     → sort left-to-right)
                                     │
                                     ▼
                          preprocessing.py
                (same normalization used at train time)
                                     │
                                     ▼
                          inference.py (ONNX Runtime)
                    per-crop classification + confidence
                                     │
                                     ▼
                    assembled string + per-char JSON
                                     │
                                     ▼
                     browser renders result, confidence
                     bars, and segmented crops
```

## Project layout

```
handwriting-recognizer/
├── model/
│   ├── dataset.py          # EMNIST loading + label mapping
│   ├── train.py             # trains the CNN, saves checkpoint
│   └── export_onnx.py       # converts checkpoint -> model.onnx
├── backend/
│   ├── app.py                # FastAPI routes
│   ├── segmentation.py       # connected-component character segmentation
│   ├── preprocessing.py      # shared train/serve normalization
│   ├── inference.py          # ONNX Runtime wrapper + Grad-CAM
│   └── requirements.txt
├── frontend/
│   ├── index.html         # landing page
│   ├── app.html           # the drawing app itself
│   ├── style.css
│   └── script.js
├── Dockerfile
└── README.md
```

## Build order (recommended)

1. **Train the model** (`model/train.py`) on EMNIST Balanced (47 classes:
   digits + unambiguous letters). Confirm standalone test accuracy before
   touching serving code.
2. **Export to ONNX** (`model/export_onnx.py`).
3. **Test segmentation** on a few saved canvas PNGs, independent of the API —
   this is the part most likely to need iteration (dotted `i`/`j`, touching
   characters, stray marks).
4. **Wire segmentation → preprocessing → inference** in a plain script, check
   it produces sane strings.
5. **Run the FastAPI backend**, hit `/predict` with `curl` before touching
   the frontend.
6. **Open `frontend/index.html`** (the landing page), then click through to
   `app.html` to draw and predict.

## Running it

### 1. Train

```bash
cd model
pip install -r ../backend/requirements.txt torch torchvision
python train.py --epochs 8 --out ../backend/checkpoint.pt
python export_onnx.py --checkpoint ../backend/checkpoint.pt --out ../backend/model.onnx
```

EMNIST downloads automatically via `torchvision.datasets.EMNIST` on first run
(~500MB, requires network access).

### 2. Serve

```bash
cd backend
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

### 3. Frontend

Two ways to run it locally:

- **Open the HTML files directly** (`frontend/index.html`, then `app.html`)
  as `file://` URLs - `script.js` detects this and points API calls at
  `http://localhost:8000` automatically.
- **Or just visit the backend itself** at `http://localhost:8000/` once
  it's running (see step 2) - `app.py` serves the frontend directly, so
  there's no separate frontend server needed at all, and API calls are
  same-origin.

### 4. Docker (single service - frontend + API together)

```bash
docker build -t blidscribe .
docker run -p 8000:8000 blidscribe
```

Visit `http://localhost:8000/` for the landing page, or
`http://localhost:8000/app.html` for the app directly. `model.onnx`,
`model_classes.json`, and `checkpoint.pt` must exist in `backend/` before
building the image (see step 1) - the Dockerfile copies `backend/`,
`frontend/`, and the two `model/` scripts needed for `/explain` into one
image.

## Deploying

The recommended setup is **one deployed service**, not a separate
frontend host + backend host: `app.py` already serves the frontend as
static files (see `FRONTEND_DIR` / the `StaticFiles` mount at the bottom
of `backend/app.py`), so the Docker image built above is a complete,
self-contained deployable unit - no CORS to configure, no second hosting
target, no `API_BASE` to point at a different domain.

**To deploy:**

1. Make sure `backend/model.onnx`, `backend/model_classes.json`, and
   `backend/checkpoint.pt` exist and are committed/included in whatever
   you deploy (they're build artifacts, so they're easy to accidentally
   `.gitignore` - double-check before pushing).
2. Push this repo (with the Dockerfile at the root) to any platform that
   builds from a Dockerfile and gives you a public HTTPS URL - Render,
   Railway, Fly.io, and Google Cloud Run are all straightforward fits for
   a project this size, and typically have a free/cheap tier sufficient
   for this model's traffic and compute needs.
3. Those platforms usually inject a `$PORT` environment variable and
   expect the container to bind to it - the Dockerfile's `CMD` already
   handles this (`--port ${PORT:-8000}`), so no extra config is needed.
4. That's it - once deployed, the platform's URL serves both the landing
   page and the app, and `script.js` talks to the API on the same origin
   automatically.

**If you'd rather host the frontend separately** (e.g. a CDN like Vercel
or GitHub Pages for the static files, with the backend elsewhere), two
changes are required instead of the above:
- In `frontend/script.js`, change `API_BASE` from the same-origin default
  to the backend's actual public URL (e.g.
  `const API_BASE = "https://your-backend.example.com";`).
- In `backend/app.py`, tighten the CORS `allow_origins` from `["*"]` to
  the frontend's exact deployed domain, since the two are no longer
  same-origin.
- Both the frontend and backend need HTTPS in this setup - a frontend
  served over HTTPS calling an `http://` backend will be blocked by the
  browser as mixed content.

**A note on cost/size:** `torch` (needed only for the `/explain` Grad-CAM
endpoint) is by far the heaviest dependency in `backend/requirements.txt`.
If your hosting platform bills by image size or memory, and you're
willing to drop the Grad-CAM feature, removing `torch`/`torchvision` from
`requirements.txt` and deleting the `/explain` route in `app.py` shrinks
the image substantially - `/predict` only ever needed `onnxruntime`.

## Design notes / known limitations

- **EMNIST Balanced** merges visually-similar upper/lowercase pairs (e.g.
  `C`/`c`, `O`/`o`, `S`/`s`) into one class to keep the label space sane —
  the model genuinely cannot distinguish case for those letters, and the API
  reports whichever canonical form the dataset uses. This is documented in
  `model/dataset.py`.
- **Segmentation** merges bounding boxes that are horizontally overlapping
  and vertically stacked, to handle `i`/`j` dots and cursive-adjacent marks.
  It is a connected-component heuristic, not a learned detector — touching
  or overlapping characters (e.g. cursive joins) will under-segment. This is
  the single most likely source of errors; see `segmentation.py` docstring
  for the exact heuristic and its failure modes.
- **Train/serve skew**: EMNIST characters are centered, size-normalized, and
  anti-aliased. Raw canvas strokes are not. `preprocessing.py` is the single
  shared function used identically at training time (via `dataset.py`) and
  serving time (via `inference.py`) specifically to avoid the classic mistake
  of the model seeing different-looking inputs at train vs. inference time.

# Single-service image: serves BOTH the static frontend and the API from
# one FastAPI app (see the StaticFiles mount at the end of backend/app.py).
# This is the recommended deployment - one container, no CORS, no separate
# frontend host to manage.
FROM python:3.11-slim

WORKDIR /app

# System deps for scipy/onnxruntime wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt ./requirements.txt

# If you don't need /explain (Grad-CAM), drop the last line of
# requirements.txt (torch) before building to keep the image small -
# /predict only needs onnxruntime.
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY model/train.py ./model/train.py
COPY model/dataset.py ./model/dataset.py
COPY frontend/ ./frontend/

WORKDIR /app/backend

# model.onnx, model_classes.json, and (optionally) checkpoint.pt must be
# present in backend/ at build time - see README for how to produce them.
# They are not copied by a wildcard here so an accidental missing model
# fails loudly at startup (inference.py raises FileNotFoundError) instead
# of silently shipping a stale or absent model.

EXPOSE 8000

# Shell form (not exec/array form) so ${PORT} actually gets expanded - most
# PaaS platforms (Render, Railway, Fly.io, Cloud Run) inject a $PORT env
# var and expect the container to bind to it rather than a fixed port.
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}


FROM python:3.11-slim

WORKDIR /app

# System deps for scipy/onnxruntime wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY model/train.py ./model/train.py
COPY model/dataset.py ./model/dataset.py
COPY frontend/ ./frontend/

WORKDIR /app/backend

EXPOSE 8000

CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}


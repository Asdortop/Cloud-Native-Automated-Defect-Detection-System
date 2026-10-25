# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile — NEU Defect Inspection Serving Service
#
# Build:
#   docker build -f docker/Dockerfile -t defect-serving:latest .
#
# Run:
#   docker run --rm -p 8080:8080 \
#     -v $(pwd)/models:/app/models:ro \
#     -e ONNX_MODEL_PATH=/app/models/defect_efficientnet_b0.onnx \
#     -e CLASS_NAMES="Crazing,Inclusion,Patches,Pitted_surface,Rolled-in_scale,Scratches" \
#     defect-serving:latest
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.10-slim AS base

# Keeps Python from generating .pyc files and enables unbuffered stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8

WORKDIR /app

# ── System dependencies ────────────────────────────────────────────────────
# libgomp1  : required by onnxruntime (OpenMP)
# libglib2-0: required by Pillow / PIL for some image formats
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ────────────────────────────────────────────────────
# Copy only requirements first so Docker can cache this layer independently
# of source code changes.
COPY requirements.txt .

# Install CPU-only torch + torchvision (much smaller than the default CUDA
# wheel; the serving path only uses torchvision for the transform pipeline).
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir \
        torch==2.1.2+cpu \
        torchvision==0.16.2+cpu \
        --extra-index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir \
        fastapi \
        "uvicorn[standard]" \
        onnxruntime \
        Pillow \
        numpy \
        prometheus-fastapi-instrumentator \
        kafka-python \
        psycopg2-binary

# ── Application source ─────────────────────────────────────────────────────
COPY src/ ./src/

# Ensure the src package is importable as a top-level package
ENV PYTHONPATH=/app

# ── Model volume ───────────────────────────────────────────────────────────
# The ONNX model is mounted at runtime via -v so the image stays model-agnostic.
# Create the default directory so the path exists even without a volume.
RUN mkdir -p /app/models

# ── Runtime configuration ──────────────────────────────────────────────────
ENV ONNX_MODEL_PATH=/app/models/defect_efficientnet_b0.onnx \
    CLASS_NAMES="Crazing,Inclusion,Patches,Pitted_surface,Rolled-in_scale,Scratches" \
    LOG_LEVEL=info \
    HOST=0.0.0.0 \
    PORT=8080

EXPOSE 8080

# ── Health check ───────────────────────────────────────────────────────────
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"

# ── Entrypoint ─────────────────────────────────────────────────────────────
CMD ["sh", "-c", \
     "uvicorn src.serving.main:app \
      --host ${HOST} \
      --port ${PORT} \
      --log-level ${LOG_LEVEL} \
      --workers 1"]

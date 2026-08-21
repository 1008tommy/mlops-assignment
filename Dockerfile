# syntax=docker/dockerfile:1
#
# IT3385 MLOps web app — container image for GCP Cloud Run.
#
# Multi-stage build:
#   Stage 1 (builder)  — install project dependencies with Poetry into a venv
#   Stage 2 (runtime)  — slim Python image + the venv + the web app + models/data
#
# Build context is the repo working tree, so the model .pkl files and raw data
# CSVs are bundled into the image even though they are git-ignored / DVC-tracked.
# Run `dvc pull` locally first so all data files exist before building.

# ---------------------------------------------------------------------------
# Stage 1: dependencies
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

ENV POETRY_VERSION=2.4.1 \
    POETRY_HOME=/opt/poetry \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1

RUN pip install --no-cache-dir "poetry==${POETRY_VERSION}"

WORKDIR /app
COPY pyproject.toml poetry.lock ./
RUN poetry install --only main --no-root

# ---------------------------------------------------------------------------
# Stage 2: runtime
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# libgomp (GNU OpenMP) is required by lightgbm/xgboost at import time
# (loading the PyCaret model pulls in lightgbm) but is not in the slim image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Poetry-managed virtualenv with all runtime dependencies
COPY --from=builder /app/.venv /app/.venv

# Web app, models, and data (the views read these paths at runtime)
COPY webapp/ webapp/
COPY javian/ javian/
COPY darren/ darren/
COPY .streamlit/ .streamlit/

EXPOSE 8080

# Streamlit health endpoint — used for local `docker run`, ignored by Cloud Run
HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8080/_stcore/health', timeout=3)"]

# Headless + port come from .streamlit/config.toml
CMD ["streamlit", "run", "webapp/streamlit_app.py"]

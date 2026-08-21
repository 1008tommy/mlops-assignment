# syntax=docker/dockerfile:1
#
# IT3385 MLOps web app — container image for GCP Cloud Run.
#
# Multi-stage build:
#   Stage 1 (builder)  — install dependencies with Poetry, then `dvc pull` the
#                        DVC-tracked data (raw CSVs) from the gdrive remote
#   Stage 2 (runtime)  — slim Python image + venv + web app + models + data
#
# DVC credentials are injected at build time via the GDRIVE_CREDENTIALS_DATA
# build arg (from a Secret Manager secret in Cloud Build). When the arg is
# empty the pull is skipped, so local builds against the working tree work
# without credentials.

# ---------------------------------------------------------------------------
# Stage 1: dependencies + data
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

ENV POETRY_VERSION=2.4.1 \
    POETRY_HOME=/opt/poetry \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    PATH="/app/.venv/bin:$PATH"

RUN pip install --no-cache-dir "poetry==${POETRY_VERSION}"

WORKDIR /app
COPY pyproject.toml poetry.lock ./
RUN poetry install --only main --no-root

# Web app source, models, and DVC config/pointers (all tracked in git)
COPY webapp/ webapp/
COPY javian/ javian/
COPY darren/ darren/
COPY .dvc/ .dvc/
COPY .dvcignore ./

# Pull the DVC-tracked data (raw CSVs) from the gdrive remote.
# Skipped when GDRIVE_CREDENTIALS_DATA is empty (e.g. local builds).
ARG GDRIVE_CREDENTIALS_DATA
RUN if [ -n "$GDRIVE_CREDENTIALS_DATA" ]; then \
      GDRIVE_CREDENTIALS_DATA="$GDRIVE_CREDENTIALS_DATA" dvc pull; \
    fi

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

# Everything for runtime comes from the builder, so the dvc-pulled data is
# included. .streamlit/ is copied from the build context.
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/webapp /app/webapp
COPY --from=builder /app/javian /app/javian
COPY --from=builder /app/darren /app/darren
COPY .streamlit/ .streamlit/

EXPOSE 8080

# Streamlit health endpoint — used for local `docker run`, ignored by Cloud Run
HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8080/_stcore/health', timeout=3)"]

# Headless + port come from .streamlit/config.toml
CMD ["streamlit", "run", "webapp/streamlit_app.py"]

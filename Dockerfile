# syntax=docker/dockerfile:1
# IT3385 MLOps — full-MLOps image (pull data -> train -> serve)
#
# Two entrypoints:
#   serve (default):  Streamlit web app
#   train:            dvc pull + run the training pipeline (see dvc.yaml)
#
# Example:
#   docker build -t mlops-app .
#   docker run -p 8080:8080 mlops-app                              # serve
#   docker run mlops-app python javian/src/main.py                 # train (stub today)
#
# NOTE: DVC uses a Google Drive remote (see .dvc/config). To `dvc pull` inside
# the container you must provide GDrive credentials via the
# GDRIVE_CREDENTIALS_DATA env var (a Google service-account JSON string).

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VERSION=2.0.1

# System deps: git (DVC needs it), build tools (some wheels compile from source)
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Poetry (dependency manager), matching the repo's pyproject.toml / poetry.lock
RUN pip install "poetry==${POETRY_VERSION}"

WORKDIR /app

# Install project dependencies in a cacheable layer (only re-runs when
# pyproject.toml / poetry.lock change). package-mode=false means Poetry won't
# try to install the app itself, so --no-root is not required.
COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create false \
    && poetry install --without dev --no-root

# App code last, so code edits don't invalidate the dependency cache layer
COPY . .

EXPOSE 8080

# Default: serve the Streamlit web app.
# Cloud Run injects $PORT (default 8080); Streamlit must bind 0.0.0.0 + headless.
CMD ["streamlit", "run", "webapp/streamlit_app.py", \
     "--server.port=8080", "--server.address=0.0.0.0", "--server.headless=true"]

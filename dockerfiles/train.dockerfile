FROM ghcr.io/astral-sh/uv:python3.12-bookworm AS base

# Install system dependencies required for building native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    gfortran \
    build-essential \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy lock and project metadata first (cache friendly)
COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
# Copy README (required by pyproject metadata)
COPY README.md README.md

# Resolve dependencies (no project install yet)
RUN uv sync --frozen --no-install-project

# Copy sources and any other packaging files the build expects
COPY src src/

# Install project into the environment
RUN uv sync --frozen

ENTRYPOINT ["uv", "run", "src/dtu_mlops_project/train.py"]

FROM ghcr.io/astral-sh/uv:python3.12-bookworm AS base

# Install system dependencies required for scientific Python builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    gfortran \
    build-essential \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml

# Resolve dependencies (no project install yet)
RUN uv sync --frozen --no-install-project

COPY src src/

# Install project
RUN uv sync --frozen

ENTRYPOINT ["uv", "run", "src/dtu_mlops_project/train.py"]

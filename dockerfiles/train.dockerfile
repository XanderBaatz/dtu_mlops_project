FROM ghcr.io/astral-sh/uv:python3.12-bookworm AS base

# Install system dependencies needed for native builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    gfortran build-essential pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy metadata and packaging files first (cache friendly)
COPY pyproject.toml pyproject.toml
COPY uv.lock uv.lock
# Ensure files referenced in pyproject.toml are copied
COPY README.md README.md
COPY LICENSE LICENSE
# If your pyproject references other files (MIT, CHANGELOG.md...), copy them too:
# COPY MIT MIT
# COPY CHANGELOG.md CHANGELOG.md

# Resolve and download dependency wheels / sdist (no project install)
RUN uv sync --frozen --no-install-project

# Copy source
COPY src src/

# Install project into the environment
RUN uv sync --frozen

ENTRYPOINT ["uv", "run", "src/dtu_mlops_project/train.py"]

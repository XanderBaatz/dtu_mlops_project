FROM ghcr.io/astral-sh/uv:python3.12-bookworm AS base

# Install system dependencies needed for native builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    gfortran build-essential pkg-config curl \
    && rm -rf /var/lib/apt/lists/*

# Copy metadata and packaging files first (cache friendly)
COPY pyproject.toml pyproject.toml
COPY uv.lock uv.lock
# Ensure files referenced in pyproject.toml are copied
COPY README.md README.md
COPY LICENSE LICENSE
COPY .project-root .project-root
COPY configs/ configs/
COPY src/ src/
# If your pyproject references other files (MIT, CHANGELOG.md...), copy them too:
# COPY MIT MIT
# COPY CHANGELOG.md CHANGELOG.md

# INEFFICIENT VERSION
# Resolve and download dependency wheels / sdist (no project install)
#RUN uv sync --frozen --no-install-project

WORKDIR /

## OPTIMIZED VERSION
# Below is an optimized version that uses caching of uv downloads/wheels between builds
# to speed up the build process while still keeping the final image small.
# This makes sure we don't always rebuild everything from scratch.
ENV UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache/uv uv sync

# Create directories for saving outputs
RUN mkdir -p models reports/figures logs/train/runs profiler

# Install project into the environment
#RUN uv sync --frozen

ENTRYPOINT ["uv", "run", "src/dtu_mlops_project/train.py"]

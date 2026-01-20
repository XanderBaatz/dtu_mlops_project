FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# System dependencies for native builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    gfortran \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy project metadata
COPY pyproject.toml pyproject.toml
COPY uv.lock uv.lock
COPY src/ src/
COPY configs/ configs/
COPY tests/ tests/
COPY README.md README.md
COPY LICENSE LICENSE
COPY .project-root .project-root
COPY .python-version .python-version

# Set working directory
WORKDIR /

# Install python and dependencies via uv
ENV UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache/uv uv sync

RUN mkdir -p models reports/figures logs profiler

# Set default entrypoint to training script
ENTRYPOINT ["uv", "run", "src/dtu_mlops_project/train.py"]

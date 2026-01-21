# Multi stage uv
FROM ghcr.io/astral-sh/uv:latest AS uv

# Use Python 3.12 Bookworm base image (amd64)
FROM mcr.microsoft.com/devcontainers/python:3.12-bookworm

# Copy uv from the official image
COPY --from=uv /uv /uvx /bin/

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
ENV UV_COMPILE_BYTECODE=1
RUN uv sync --no-cache-dir --frozen

RUN mkdir -p models reports/figures logs profiler

# Set default entrypoint to training script
ENTRYPOINT ["uv", "run", "src/dtu_mlops_project/train.py"]

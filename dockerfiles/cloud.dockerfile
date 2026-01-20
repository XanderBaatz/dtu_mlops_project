# Multi-stage build to get uv
FROM ghcr.io/astral-sh/uv:latest AS uv

# Base image with CUDA 12.4.1 and cuDNN (compatible with common PyTorch versions)
# Ubuntu 22.04 base
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# Copy uv from the official image
COPY --from=uv /uv /uvx /bin/

# Install essentials
# - build-essential, gcc: for compiling some python extensions if needed
# - git: often needed for dependencies
# - ca-certificates, curl: for downloading things
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gfortran \
    build-essential \
    gcc \
    git \
    ca-certificates \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml pyproject.toml
COPY uv.lock uv.lock
COPY src/ src/
COPY configs/ configs/
COPY tests/ tests/
COPY README.md README.md
COPY LICENSE LICENSE
COPY .python-version .python-version
COPY .project-root .project-root

WORKDIR /

# Install python and dependencies via uv
# uv should strictly respect .python-version and install the required python
ENV UV_COMPILE_BYTECODE=1
RUN uv sync --no-cache-dir --frozen

RUN mkdir -p models reports/figures

# Use uv run to execute the training script (project module path)
ENTRYPOINT ["uv", "run", "src/dtu_mlops_project/train.py"]

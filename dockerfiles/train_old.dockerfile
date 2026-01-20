# Use Python 3.12 Bookworm base image (amd64)
FROM mcr.microsoft.com/devcontainers/python:3.12-bookworm

# Set working directory
WORKDIR /app

# System dependencies for native builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    gfortran \
    build-essential \
    pkg-config \
    apt-transport-https \
    ca-certificates \
    gnupg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv (Astral SH)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Make uv globally available
ENV PATH="/root/.local/bin:${PATH}"

# Verify installations
RUN uv --version \
    && python --version \
    && gfortran --version

# Copy project metadata first for caching
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY .project-root .project-root

# Resolve and download dependency wheels / sdist (no project install yet)
RUN uv sync --frozen --no-install-project

# Copy source code
COPY src src/

# Copy project configs
COPY configs configs/

# Install the project into the environment
RUN uv sync --frozen

# Set default entrypoint to training script
ENTRYPOINT ["uv", "run", "python", "-m", "dtu_mlops_project.train"]

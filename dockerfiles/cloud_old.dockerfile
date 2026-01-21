# Cloud training image (CPU) similar to train.dockerfile
FROM mcr.microsoft.com/devcontainers/python:3.12-bookworm

WORKDIR /app

# System deps for native builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    gfortran \
    build-essential \
    pkg-config \
    apt-transport-https \
    ca-certificates \
    gnupg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv (Astral)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Verify
RUN uv --version && python --version && gfortran --version

# Copy project metadata first
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY .project-root .project-root

# Resolve deps (no project install yet)
# Skip frozen lockfile due to PyTorch CPU mirror availability issues
RUN uv sync --no-install-project || \
    (echo "Frozen sync failed, trying without frozen..." && uv sync --no-install-project --no-lock)

# Copy source and configs
COPY src/ src/
COPY configs/ configs/

# Verify configs directory structure (force rebuild)
RUN echo "=== Config directory structure ===" && \
    ls -la configs/ && \
    echo "=== All config files ===" && \
    find configs -type f -name "*.yaml" && \
    echo "=== Data config check ===" && \
    ls -la configs/data/

# Install project
RUN uv sync || \
    (echo "Frozen sync failed, trying without frozen..." && uv sync --no-lock)

# Entrypoint
ENTRYPOINT ["uv", "run", "python", "-m", "dtu_mlops_project.train"]

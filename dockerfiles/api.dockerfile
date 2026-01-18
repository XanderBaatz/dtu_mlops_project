FROM ghcr.io/astral-sh/uv:python3.12-bookworm AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    gfortran \
    build-essential \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml

RUN uv sync --frozen --no-install-project

COPY src src/

RUN uv sync --frozen

ENTRYPOINT ["uv", "run", "uvicorn", "src.dtu_mlops_project.api:app", "--host", "0.0.0.0", "--port", "8000"]

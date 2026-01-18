FROM ghcr.io/astral-sh/uv:python3.12-bookworm AS base

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml

# Resolve dependencies (no project install yet)
RUN uv sync --frozen --no-install-project

COPY src src/

# Install project into the environment
RUN uv sync --frozen

ENTRYPOINT ["uv", "run", "src/dtu_mlops_project/train.py"]

#!/usr/bin/env bash
set -e


apt-get update && apt-get install -y \
    gfortran \
    build-essential \
    pkg-config


curl -LsSf https://astral.sh/uv/install.sh | sh

# Ensure uv is available
export PATH="$HOME/.local/bin:$PATH"

# Cargo env
if [ -f "$HOME/.cargo/env" ]; then
    source "$HOME/.cargo/env"
fi


uv --version
python --version
gfortran --version


uv sync --dev

uv run pre-commit install --install-hooks

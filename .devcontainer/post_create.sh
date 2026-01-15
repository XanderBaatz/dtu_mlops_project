#!/usr/bin/env bash
set -e

# Install system dependencies
apt-get update && apt-get install -y gfortran build-essential pkg-config

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add uv to PATH
export PATH="$HOME/.local/bin:$PATH"

# OPTIONAL: only source Cargo if it exists
if [ -f "$HOME/.cargo/env" ]; then
    source "$HOME/.cargo/env"
fi

# Verify installations
uv --version
python --version
gfortran --version

# Install project dependencies
uv sync --dev

# Install pre-commit hooks
uv run pre-commit install --install-hooks

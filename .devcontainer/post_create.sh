#! /usr/bin/env bash

set -e

# Install system dependencies
sudo apt-get update && sudo apt-get install -y --no-install-recommends \
    gfortran \
    build-essential \
    ca-certificates \
    curl \
    && sudo apt-get clean && sudo rm -rf /var/lib/apt/lists/*

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
#source $HOME/.cargo/env

# Install Dependencies
uv sync --dev

# Install pre-commit hooks
uv run pre-commit install --install-hooks

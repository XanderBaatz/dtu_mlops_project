#! /usr/bin/env bash

# Install system dependencies
apt-get update && apt-get install -y gfortran

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env

# Install Dependencies
uv sync --dev

# Install pre-commit hooks
uv run pre-commit install --install-hooks

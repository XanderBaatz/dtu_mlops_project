#!/usr/bin/env bash
set -e

# System dependencies

sudo apt-get update
sudo apt-get install -y \
    gfortran \
    build-essential \
    pkg-config \
    apt-transport-https \
    ca-certificates \
    gnupg \
    curl


# Google Cloud SDK

if ! command -v gcloud >/dev/null 2>&1; then
    echo "Installing Google Cloud SDK..."

    sudo mkdir -p /etc/apt/keyrings

    curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
        | sudo gpg --dearmor -o /etc/apt/keyrings/cloud.google.gpg

    echo "deb [signed-by=/etc/apt/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
        | sudo tee /etc/apt/sources.list.d/google-cloud-sdk.list > /dev/null

    sudo apt-get update
    sudo apt-get install -y google-cloud-sdk
fi


# uv

curl -LsSf https://astral.sh/uv/install.sh | sh

# Ensure uv is available
export PATH="$HOME/.local/bin:$PATH"


# Cargo (if present)

if [ -f "$HOME/.cargo/env" ]; then
    source "$HOME/.cargo/env"
fi


# Verify installs

uv --version
python --version
gfortran --version
gcloud --version


# Project setup

uv sync --dev
uv run pre-commit install --install-hooks

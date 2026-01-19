"""Tests for the FastAPI application."""

from io import BytesIO
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from dtu_mlops_project.apifile import app


@pytest.fixture(autouse=True)
def clear_model_cache():
    """Ensure model cache is cleared between tests."""

    from dtu_mlops_project import apifile

    apifile._load_model.cache_clear()
    yield
    apifile._load_model.cache_clear()


@pytest.fixture
def client():
    """Fixture for the test client."""
    return TestClient(app)


def test_health_endpoint(client):
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_endpoint_with_valid_image(client):
    """Test predict endpoint with a valid image."""
    img = Image.new("RGB", (100, 100), color="white")
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)

    with patch("dtu_mlops_project.apifile._load_model") as mock_model:
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance

        import torch

        mock_model_instance.return_value = torch.tensor(
            [[0.1, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        )

        response = client.post("/predict", files={"file": ("test.png", img_byte_arr, "image/png")})

    assert response.status_code == 200
    data = response.json()
    assert "class_id" in data
    assert "class_name" in data
    assert "confidence" in data


def test_predict_endpoint_with_real_model_and_sample_image(client):
    """Integration test using the real model and a repo image."""

    sample_image = Path(__file__).resolve().parent.parent / "tests" / "testimage.jpg"
    if not sample_image.exists():
        pytest.skip("Sample image not available in repository")

    model_path = Path("models/model.pth")
    if not model_path.exists():
        pytest.skip("Model checkpoint missing at models/model.pth")

    with sample_image.open("rb") as img_file:
        response = client.post(
            "/predict",
            files={"file": (sample_image.name, img_file, "image/jpeg")},
        )

    assert response.status_code == 200
    data = response.json()
    assert "class_id" in data
    assert "class_name" in data
    assert "confidence" in data
    assert isinstance(data["class_id"], int)
    assert isinstance(data["class_name"], str)
    assert isinstance(data["confidence"], float)
    assert 0 <= data["class_id"] < 10
    assert 0 <= data["confidence"] <= 1


def test_predict_endpoint_with_invalid_image(client):
    """Test predict endpoint with invalid image data."""
    invalid_data = BytesIO(b"not an image")

    response = client.post("/predict", files={"file": ("invalid.txt", invalid_data, "text/plain")})

    assert response.status_code == 400
    assert "Invalid image file" in response.json()["detail"]


def test_predict_endpoint_model_not_found(client):
    """Test predict endpoint when model file is missing."""
    img = Image.new("RGB", (100, 100), color="white")
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)

    with patch("dtu_mlops_project.apifile._load_model") as mock_model:
        mock_model.side_effect = FileNotFoundError("Model not found")

        response = client.post("/predict", files={"file": ("test.png", img_byte_arr, "image/png")})

    assert response.status_code == 500
    assert "Model checkpoint not found" in response.json()["detail"]

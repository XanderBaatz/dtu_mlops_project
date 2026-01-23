"""FastAPI app that predicts Fashion-MNIST class from an uploaded image."""

from functools import lru_cache, partial
from io import BytesIO
import os
from pathlib import Path
from time import time
from types import SimpleNamespace
from typing import Any, Dict

import requests

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse
from PIL import Image, UnidentifiedImageError
import torch
import torchvision.transforms as transforms
from prometheus_client import Counter, Histogram, Summary, generate_latest

from dtu_mlops_project.model import CNN, C8SteerableCNN


app = FastAPI(title="Fashion MNIST Classifier")

# Prometheus metrics
api_requests_total = Counter(
    "api_requests_total",
    "Total number of API requests received",
    ["method", "endpoint"],
)
api_errors_total = Counter(
    "api_errors_total",
    "Total number of API errors",
    ["endpoint", "type"],
)
classification_duration = Histogram(
    "classification_duration_seconds",
    "Time spent classifying a review",
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)
review_size_summary = Summary(
    "review_size_bytes",
    "Size of reviews processed, in bytes",
)

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/model.pth"))
MODEL_TYPE = os.getenv("MODEL_TYPE", "cnn").lower()
FASHION_MNIST_LABELS = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

_transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)


def _net_config() -> SimpleNamespace:
    """Return namespace matching CNN `net` hyperparameters."""
    return SimpleNamespace(
        input_channels=1,
        kernel_size=3,
        padding=1,
        pooling_size=2,
        num_classes=len(FASHION_MNIST_LABELS),
    )


def _net_config_c8() -> SimpleNamespace:
    """Return namespace matching C8SteerableCNN `net` hyperparameters."""
    return SimpleNamespace(
        input_channels=1,
        kernel_size=3,
        padding=1,
        pooling_size=2,
        stride=1,
        num_classes=len(FASHION_MNIST_LABELS),
    )


@lru_cache(maxsize=1)
def _load_model() -> torch.nn.Module:
    """Load and cache the trained model."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {MODEL_PATH}")

    if MODEL_TYPE == "c8":
        model: torch.nn.Module = C8SteerableCNN(
            net=_net_config_c8(),  # type: ignore[arg-type]
            optimizer=partial(torch.optim.Adam, lr=1e-3),  # type: ignore[arg-type]
        )
    else:
        model = CNN(
            net=_net_config(),  # type: ignore[arg-type]
            optimizer=partial(torch.optim.Adam, lr=1e-3),  # type: ignore[arg-type]
        )

    checkpoint = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


@app.get("/health")
def health() -> Dict[str, str]:
    """Health probe endpoint."""
    api_requests_total.labels(method="GET", endpoint="/health").inc()
    return {"status": "ok"}


@app.get("/metrics")
def metrics() -> PlainTextResponse:
    """Expose Prometheus metrics."""
    api_requests_total.labels(method="GET", endpoint="/metrics").inc()
    return PlainTextResponse(generate_latest())


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> JSONResponse:
    """Predict Fashion-MNIST class from an uploaded image file."""
    api_requests_total.labels(method="POST", endpoint="/predict").inc()
    start_time = time()

    try:
        contents = await file.read()
        image: Image.Image = Image.open(BytesIO(contents))
    except UnidentifiedImageError as exc:
        api_errors_total.labels(endpoint="/predict", type="invalid_image").inc()
        raise HTTPException(status_code=400, detail="Invalid image file.") from exc

    image = image.convert("L")
    tensor = _transform(image).unsqueeze(0)

    try:
        model = _load_model()
    except FileNotFoundError as exc:
        api_errors_total.labels(endpoint="/predict", type="model_missing").inc()
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    class_id = int(torch.argmax(probabilities).item())
    confidence = float(probabilities[class_id].item())

    # Record metrics
    class_name = FASHION_MNIST_LABELS[class_id]
    classification_duration.observe(time() - start_time)
    review_size_summary.observe(len(contents))

    payload: Dict[str, Any] = {
        "class_id": class_id,
        "class_name": class_name,
        "confidence": confidence,
    }
    return JSONResponse(content=payload)


def main() -> None:
    base_url = os.getenv("API_BASE_URL", "http://127.0.0.1:8001")
    image_path = Path(os.getenv("IMAGE_PATH", "data/FashionMNIST/raw/image.png"))

    health_response = requests.get(f"{base_url}/health", timeout=10)
    print({"endpoint": "/health", "status_code": health_response.status_code, "body": health_response.json()})

    if not image_path.exists():
        raise FileNotFoundError(f"Image path not found: {image_path}")

    with image_path.open("rb") as file_handle:
        predict_response = requests.post(f"{base_url}/predict", files={"file": file_handle}, timeout=30)
    print(
        {
            "endpoint": "/predict",
            "status_code": predict_response.status_code,
            "body": predict_response.json(),
        }
    )

    metrics_response = requests.get(f"{base_url}/metrics", timeout=10)
    print({"endpoint": "/metrics", "status_code": metrics_response.status_code})
    print(metrics_response.text)


if __name__ == "__main__":
    main()

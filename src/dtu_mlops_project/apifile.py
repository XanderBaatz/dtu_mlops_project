"""FastAPI app that predicts Fashion-MNIST class from an uploaded image."""

from functools import lru_cache, partial
from io import BytesIO
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
import torch
import torchvision.transforms as transforms

from dtu_mlops_project.model import CNN


app = FastAPI(title="Fashion MNIST Classifier")

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/model.pth"))
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


@lru_cache(maxsize=1)
def _load_model() -> torch.nn.Module:
    """Load and cache the trained model."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {MODEL_PATH}")

    model = CNN(net=_net_config(), optimizer=partial(torch.optim.Adam, lr=1e-3))
    checkpoint = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


@app.get("/health")
def health() -> Dict[str, str]:
    """Health probe endpoint."""

    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> JSONResponse:
    """Predict Fashion-MNIST class from an uploaded image file."""

    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents))
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Invalid image file.") from exc

    image = image.convert("L")
    tensor = _transform(image).unsqueeze(0)

    try:
        model = _load_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    class_id = int(torch.argmax(probabilities).item())
    confidence = float(probabilities[class_id].item())

    payload: Dict[str, Any] = {
        "class_id": class_id,
        "class_name": FASHION_MNIST_LABELS[class_id],
        "confidence": confidence,
    }
    return JSONResponse(content=payload)

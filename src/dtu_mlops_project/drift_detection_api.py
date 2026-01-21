from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pathlib import Path
from evidently.legacy.report import Report
from evidently.legacy.metric_preset import DataDriftPreset

from tests.data_drift_detection import extract_features, load_images

app = FastAPI(title="Fashion-MNIST Drift Monitor")

DATA_DIR = Path("./data/fashion_mnist/raw")
REPORT_PATH = Path("drift_report.html")

X_train = load_images(DATA_DIR / "train-images-idx3-ubyte")
reference_features = extract_features(X_train[:30000])

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/run_drift")
def run_drift(current_batch_size: int = 1000):
    X_test = load_images(DATA_DIR / "t10k-images-idx3-ubyte")
    current_features = extract_features(X_test[:current_batch_size])

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_features, current_data=current_features)
    report.save_html(str(REPORT_PATH))

    return JSONResponse({"status": "ok", "report_file": str(REPORT_PATH)})

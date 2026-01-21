from data_drift_detection import load_images, load_labels, extract_features
import requests
import pandas as pd
from PIL import Image
import numpy as np
from io import BytesIO

if __name__ == "__main__":
    X_test = load_images("./data/fashion_mnist/raw/t10k-images-idx3-ubyte")
    y_test = load_labels("./data/fashion_mnist/raw/t10k-labels-idx1-ubyte")

    images_to_send = X_test[:100]

    predictions = []

    API_URL = "http://localhost:8000/predict"

    for img_array in images_to_send:
        img = Image.fromarray(img_array.reshape(28,28).astype(np.uint8))

        buffered = BytesIO()
        img.save(buffered, format="PNG")
        buffered.seek(0)
        files = {"file": ("image.png", buffered, "image/png")}

        response = requests.post(API_URL, files=files)
        response.raise_for_status()
        data = response.json()
        predictions.append(data["class_id"])

    features_df = extract_features(images_to_send)
    features_df["prediction"] = predictions

    features_df.to_csv("production_input_output.csv", index=False)
    print("Saved input-output data for monitoring.")
import numpy as np
import pandas as pd
from evidently.legacy.report import Report
from evidently.legacy.metric_preset import DataDriftPreset


def load_images(path):
    with open(path, "rb") as f:
        magic, num, rows, cols = np.frombuffer(f.read(16), dtype=">u4")
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num, rows * cols)
    return images


def load_labels(path):
    with open(path, "rb") as f:
        magic, num = np.frombuffer(f.read(8), dtype=">u4")
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels


def extract_features(X, eps=1e-8):
    std_pixel = X.std(axis=1)
    std_pixel += eps  # avoid divide by zero
    return pd.DataFrame(
        {
            "mean_pixel": X.mean(axis=1),
            "std_pixel": std_pixel,
            "min_pixel": X.min(axis=1),
            "max_pixel": X.max(axis=1),
            "nonzero_ratio": (X > 0).mean(axis=1),
        }
    )


def remove_constant_columns(df):
    return df.loc[:, df.std(axis=0) > 0]


if __name__ == "__main__":
    X_train = load_images("./data/fashion_mnist/raw/train-images-idx3-ubyte")
    y_train = load_labels("./data/fashion_mnist/raw/train-labels-idx1-ubyte")

    reference = X_train[:30000]
    current = X_train[30000:]

    ref_df = pd.DataFrame(reference)
    cur_df = pd.DataFrame(current)

    ref_df = extract_features(reference)
    cur_df = extract_features(current)

    ref_df = remove_constant_columns(ref_df)
    cur_df = remove_constant_columns(cur_df)

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_df, current_data=cur_df)
    report.save_html("report.html")

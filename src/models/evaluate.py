from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error

from src.constants import DATA_DIR, METRICS_DIR
from src.models.loaders import load_xy


def main(saved_model: Path,
         X_test_path: Path,
         y_test_path: Path,
         scores: Path = METRICS_DIR / "scores.json",
         output: Path = DATA_DIR / "predictions.csv"):

    # Loading the trained model
    model = joblib.load(saved_model)

    # Loading the test data
    X_test, y_test = load_xy(X_test_path, y_test_path)

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics = {
        "mse": float(mse),
        "rmse": rmse,
        "mae": float(mae),
        "r2": float(r2),
        "n_test": int(len(y_test)),
    }

    # Save metrics
    with scores.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    # Build a dataframe with features, true labels and predictions
    df = X_test.copy()
    df["true_label"] = y_test
    df["predicted_label"] = y_pred
    df.to_csv(output, index=False)

    print("Saved predictions to:", output)
    print("Saved metrics to:", scores)
    print(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the trained model on the test set.")
    parser.add_argument(
        "--model_path",
        type=Path,
        default=DATA_DIR / "model.pkl",
        help="Path to the saved trained model.",
    )
    parser.add_argument(
        "--X_test_path",
        type=Path,
        default=DATA_DIR / "X_test.csv",
        help="Path to the test features.",
    )
    parser.add_argument(
        "--y_test_path",
        type=Path,
        default=DATA_DIR / "y_test.csv",
        help="Path to the test targets.",
    )
    parser.add_argument(
        "--scores",
        type=Path,
        default=METRICS_DIR / "scores.json",
        help="Path to save the evaluation scores.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DATA_DIR / "predictions.csv",
        help="Path to save the predictions with features and true labels.",
    )
    args = parser.parse_args()
    main(
        saved_model=args.model_path,
        X_test_path=args.X_test_path,
        y_test_path=args.y_test_path,
        scores=args.scores,
        output=args.output,
    )

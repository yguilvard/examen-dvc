import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
from src.constants import DATA_PROCESSED_DIR, MODELS_DIR
from pathlib import Path
import argparse


def main(X_train_path: Path,
         X_test_path: Path,
         output_dir: Path = DATA_PROCESSED_DIR,
         model_dir: Path = MODELS_DIR):

    os.makedirs(model_dir, exist_ok=True)

    # Load train / test features
    X_train = pd.read_csv(X_train_path)
    X_test = pd.read_csv(X_test_path)

    # Initialize scaler
    scaler = StandardScaler()

    # Fit
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert back to DataFrame (keep column names)
    X_train_scaled = pd.DataFrame(
        X_train_scaled,
        columns=X_train.columns,
    )
    X_test_scaled = pd.DataFrame(
        X_test_scaled,
        columns=X_test.columns,
    )

    # Save scaled datasets
    X_train_scaled.to_csv(output_dir/"X_train_scaled.csv", index=False)
    X_test_scaled.to_csv(output_dir/"X_test_scaled.csv", index=False)

    # Save scaler
    joblib.dump(scaler, model_dir/"scaler.joblib")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Normalize features using StandardScaler.")
    parser.add_argument(
        "--X_train_path",
        type=Path,
        default=DATA_PROCESSED_DIR / "X_train.csv",
        help="Path to the training features.",
    )
    parser.add_argument(
        "--X_test_path",
        type=Path,
        default=DATA_PROCESSED_DIR / "X_test.csv",
        help="Path to the testing features.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DATA_PROCESSED_DIR,
        help="Directory to save the normalized datasets.",
    )
    parser.add_argument(
        "--model_dir",
        type=Path,
        default=MODELS_DIR,
        help="Directory to save the scaler model.",
    )
    args = parser.parse_args()
    main(X_train_path=args.X_train_path,
         X_test_path=args.X_test_path,
         output_dir=args.output_dir,
         model_dir=args.model_dir)

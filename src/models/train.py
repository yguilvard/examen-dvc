import argparse
from pathlib import Path

import joblib

from src.constants import DATA_PROCESSED_DIR, MODELS_DIR
from src.models.loaders import load_json, load_xy
from src.models.registry import build_model


def main(config: Path, paramfile: Path, X_train_path: Path, y_train_path: Path):

    # Load config (for model_name + base kwargs)
    cfg = load_json(config)

    # Retrieve the model name from the configuration
    model_name = str(cfg.get("model_name", "")).strip()
    if not model_name:
        raise ValueError("Config error: 'model_name' is required")

    # Retrieve the kwargs used for the gridsearch
    base_kwargs = cfg.get("model_kwargs", {})

    # Load best params
    best_params = joblib.load(paramfile)

    # Merge kwargs and best params
    final_kwargs = {**base_kwargs, **best_params}

    # Build model
    model = build_model(model_name, final_kwargs)

    # Load data
    X_train, y_train = load_xy(X_train_path, y_train_path)

    # Fit
    model.fit(X_train, y_train)

    # Save trained model
    model_file = MODELS_DIR / f"{model_name}.pkl"
    joblib.dump(model, model_file)

    print("Model trained successfully")
    print("Model type:", model_name)
    print("Saved to:", model_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train final model using the identified parameters.")

    parser.add_argument("--params", default=MODELS_DIR / "/best_params.pkl")

    parser.add_argument("--config", required=True, type=Path,
                        help="Path to JSON config used in GridSearch.")
    parser.add_argument(
        "--X_train_path",
        type=Path,
        default=DATA_PROCESSED_DIR / "X_train_scaled.csv",
        help="Path to the scaled training features.",
    )
    parser.add_argument(
        "--y_train_path",
        type=Path,
        default=DATA_PROCESSED_DIR / "y_train.csv",
        help="Path to the train targets.",
    )
    args = parser.parse_args()
    main(config=args.config, paramfile=args.params,
         X_train_path=args.X_train_path, y_train_path=args.y_train_path)

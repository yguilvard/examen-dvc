from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, root_mean_squared_error, mean_absolute_error, r2_score

from src.constants import DATA_PROCESSED_DIR, METRICS_DIR, MODELS_DIR
from src.models.loaders import load_config, load_xy
from src.models.registry import build_model


SCORERS = {
    "rmse": make_scorer(root_mean_squared_error, greater_is_better=False),
    "mae": make_scorer(mean_absolute_error, greater_is_better=False),
    "r2": make_scorer(r2_score, greater_is_better=True),
}


def main(config: Path,
         X_train_path: Path,
         y_train_path: Path,
         params_output: Path,
         scoring: str = "rmse",
         cv: int = 5,
         ):

    cfg = load_config(config)

    model_name = str(cfg.get("model_name", "")).strip()
    if not model_name:
        raise ValueError("Config error: 'model_name' is required")

    model_kwargs = cfg.get("model_kwargs", {}) or {}
    param_grid = cfg.get("param_grid", {}) or {}
    if not param_grid:
        raise ValueError("Config error: 'param_grid' is empty")

    X_train, y_train = load_xy(X_train_path, y_train_path)
    model = build_model(model_name, model_kwargs)

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=SCORERS[scoring],
        cv=cv,
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X_train, y_train)

    # Get the best parameters for the model
    best_params = grid.best_params_
    best_score = grid.best_score_

    # Save best params to the expected artifact path
    params_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_params, params_output)

    results = pd.DataFrame(grid.cv_results_)
    results.to_csv(METRICS_DIR / f"{model_name}_results.csv", index=False)

    print("model:", model_name)
    print("scoring:", scoring)
    print("best_params:", best_params)
    print("best_cv_score:", best_score)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GridSearchCV regression (JSON/YAML config).")
    parser.add_argument("--config", required=True, type=Path,
                        help="Path to JSON or YAML config file.")
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
    parser.add_argument("--scoring", default="rmse",
                        choices=sorted(SCORERS.keys()))
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument(
        "--params_output",
        type=Path,
        default=MODELS_DIR / "best_params.pkl",
        help="Output path for the best params pickle.",
    )
    args = parser.parse_args()
    main(config=args.config,
         params_output=args.params_output,
         X_train_path=args.X_train_path,
         y_train_path=args.y_train_path,
         cv=args.cv,
         scoring=args.scoring)

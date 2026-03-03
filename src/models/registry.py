from typing import Any, Dict

from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from lightgbm import LGBMRegressor


MODEL_REGISTRY: Dict[str, Any] = {
    "ridge": Ridge,
    "lasso": Lasso,
    "elasticnet": ElasticNet,
    "random_forest": RandomForestRegressor,
    "gbrt": GradientBoostingRegressor,
    "lightgbm": LGBMRegressor
}


def build_model(model_name: str, model_kwargs: Dict[str, Any]):
    """Simple model factory
    Returns a "model_name" object instanciated with the given "model_kwargs"
    """
    cls = MODEL_REGISTRY[model_name]
    try:
        return cls(**(model_kwargs or {}))
    except TypeError as e:
        raise ValueError(
            f"Failed to instantiate model '{model_name}' with kwargs={model_kwargs}. Error: {e}"
        ) from e

from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_xy(X_path: Path, y_path: Path):
    X = pd.read_csv(X_path)
    y = pd.read_csv(y_path).iloc[:, 0]
    return X, y

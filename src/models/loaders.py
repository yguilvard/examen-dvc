import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_xy(X_path: Path, y_path: Path):
    X = pd.read_csv(X_path)
    y = pd.read_csv(y_path).iloc[:, 0]
    return X, y

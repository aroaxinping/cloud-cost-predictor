"""Memory quantile prediction model — parallel to the CPU model.

Trains XGBoost quantile regression on memory utilization summary features
to predict memory usage bounds for each VM.  Combined with the CPU model
in predict.py, this lets recommendations consider both dimensions:
a VM with low CPU but high predicted memory stays protected from termination.

Features (from vm_utilization_summary.csv, matching the CPU model pattern):
    mem_std   — approximated from (mem_max - mem_min) / 4 when not in CSV
    mem_min   — minimum observed memory utilization
    mem_p50   — median memory utilization
    mem_trend — slope of memory over time (0.0 when daily series unavailable)
    mem_cv    — coefficient of variation (std / mean)
"""
import csv
import logging
from pathlib import Path

import numpy as np
import xgboost as xgb
import yaml

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data" / "clean"
CONFIG_FILE = ROOT / "config.yaml"

MEM_FEATURE_NAMES = ["mem_std", "mem_min", "mem_p50", "mem_trend", "mem_cv"]


def load_config() -> dict:
    with open(CONFIG_FILE) as f:
        return yaml.safe_load(f)


def build_memory_features(csv_path: str | Path) -> tuple[list[str], np.ndarray]:
    """Build the 5 memory features from a vm_utilization_summary.csv."""
    instances: list[str] = []
    features: list[list[float]] = []

    warned_std = False
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            mem_mean = float(row.get("mem_mean", 0))
            mem_min = float(row.get("mem_min", 0))
            mem_median = float(row.get("mem_median", 0))
            mem_max = float(row.get("mem_max", 0))
            mem_n = int(row.get("mem_n", 0))

            if mem_n == 0:
                continue

            if "mem_std" in row and row["mem_std"]:
                mem_std = float(row["mem_std"])
            else:
                mem_std = (mem_max - mem_min) / 4
                if not warned_std:
                    logger.warning("mem_std not in CSV, approximating from range")
                    warned_std = True

            mem_trend = 0.0
            mem_cv = mem_std / mem_mean if mem_mean > 0 else 0

            instances.append(row["instance"])
            features.append([mem_std, mem_min, mem_median, mem_trend, mem_cv])

    return instances, np.array(features)


def load_memory_models() -> tuple[dict, list[float]] | None:
    """Load trained memory quantile models.  Returns None if not available."""
    config = load_config()
    quantiles = config["model"]["quantiles"]
    models = {}
    for q in quantiles:
        path = MODELS_DIR / f"mem_quantile_{q:.2f}.json"
        if not path.exists():
            return None
        model = xgb.XGBRegressor()
        model.load_model(path)
        models[q] = model
    return models, quantiles


def predict_memory(csv_path: str | Path) -> dict[str, dict]:
    """Run memory predictions and return {instance: {low, mid, high}}."""
    result = load_memory_models()
    if result is None:
        return {}

    models, quantiles = result
    instances, X = build_memory_features(csv_path)
    if len(instances) == 0:
        return {}

    preds = {q: models[q].predict(X) for q in quantiles}
    predictions = {}
    for i, inst in enumerate(instances):
        predictions[inst] = {
            "mem_low": float(preds[quantiles[0]][i]),
            "mem_mid": float(preds[quantiles[1]][i]),
            "mem_high": float(preds[quantiles[2]][i]),
        }
    return predictions


def train_memory_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: dict | None = None,
) -> dict:
    """Train memory quantile models (q=0.10, 0.50, 0.95).

    Returns {quantile: trained_model}.
    """
    if config is None:
        config = load_config()

    model_cfg = config["model"]
    quantiles = model_cfg["quantiles"]
    models = {}

    for q in quantiles:
        if q == 0.50:
            obj = "reg:squarederror"
        else:
            obj = "reg:quantileerror"

        params = {
            "n_estimators": model_cfg["n_estimators"],
            "max_depth": model_cfg["max_depth"],
            "learning_rate": model_cfg["learning_rate"],
            "min_child_weight": model_cfg["min_child_weight"],
            "subsample": model_cfg["subsample"],
            "colsample_bytree": model_cfg["colsample_bytree"],
            "reg_lambda": model_cfg["reg_lambda"],
            "objective": obj,
            "random_state": config.get("seed", 42),
        }
        if obj == "reg:quantileerror":
            params["quantile_alpha"] = q

        model = xgb.XGBRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        models[q] = model
        logger.info("Trained memory q=%.2f model", q)

    return models


def save_memory_models(models: dict, out_dir: Path | None = None) -> list[Path]:
    """Save trained memory models to disk."""
    if out_dir is None:
        out_dir = MODELS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for q, model in models.items():
        path = out_dir / f"mem_quantile_{q:.2f}.json"
        model.save_model(path)
        paths.append(path)
        logger.info("Saved %s", path)
    return paths

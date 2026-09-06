"""Load trained XGBoost models and generate VM recommendations from new data."""
import csv
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data" / "clean"

FEATURE_NAMES = ["std", "min", "p50", "trend", "cv"]
PRICE_PER_CPU_PCT_HOUR = 0.001
HOURS_MONTH = 730


def load_models():
    models = {}
    for q in [0.10, 0.50, 0.95]:
        path = MODELS_DIR / f"quantile_{q:.2f}.json"
        model = xgb.XGBRegressor()
        model.load_model(path)
        models[q] = model
    return models


def build_features_from_summary(csv_path):
    """Build the 5 VIF-cleaned features from a vm_utilization_summary.csv."""
    instances = []
    features = []

    with open(csv_path) as f:
        for row in csv.DictReader(f):
            cpu_mean = float(row["cpu_mean"])
            cpu_std = float(row.get("cpu_std", 0))
            cpu_min = float(row["cpu_min"])
            cpu_median = float(row["cpu_median"])
            trend = 0.0
            cv = cpu_std / cpu_mean if cpu_mean > 0 else 0

            instances.append(row["instance"])
            features.append([cpu_std, cpu_min, cpu_median, trend, cv])

    return instances, np.array(features)


def recommend(pred_high, pred_mid):
    if pred_high < 5:
        return "terminate"
    if pred_high < 20:
        return "downsize"
    if pred_mid < 50:
        return "review"
    return "keep"


def assess_risk(action, pred_high):
    if action == "terminate":
        margin = 5 - pred_high
        if margin > 3:
            return "safe"
        if margin > 1:
            return "moderate"
        return "risky"
    elif action == "downsize":
        margin = 20 - pred_high
        if margin > 8:
            return "safe"
        if margin > 3:
            return "moderate"
        return "risky"
    return "n/a"


def predict(csv_path, output_path=None):
    """Run predictions on a VM utilization summary CSV."""
    models = load_models()
    instances, X = build_features_from_summary(csv_path)

    preds = {q: models[q].predict(X) for q in [0.10, 0.50, 0.95]}

    if output_path is None:
        output_path = DATA_DIR / "vm_recommendations.csv"

    with open(output_path, "w", newline="\n") as f:
        w = csv.writer(f)
        w.writerow(["instance", "pred_low", "pred_mid", "pred_high",
                     "action", "risk", "monthly_savings"])
        for i, inst in enumerate(instances):
            low, mid, high = preds[0.10][i], preds[0.50][i], preds[0.95][i]
            action = recommend(high, mid)
            risk = assess_risk(action, high)
            savings = max(0, (X[i, 2] - mid)) * PRICE_PER_CPU_PCT_HOUR * HOURS_MONTH
            w.writerow([inst, f"{low:.2f}", f"{mid:.2f}", f"{high:.2f}",
                        action, risk, f"{savings:.2f}"])

    print(f"Wrote {len(instances)} recommendations to {output_path}")


if __name__ == "__main__":
    input_csv = sys.argv[1] if len(sys.argv) > 1 else DATA_DIR / "vm_utilization_summary.csv"
    output_csv = sys.argv[2] if len(sys.argv) > 2 else None
    predict(input_csv, output_csv)

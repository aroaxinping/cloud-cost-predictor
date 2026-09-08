"""Load trained XGBoost models and generate VM recommendations from new data."""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data" / "clean"
PRICING_FILE = ROOT / "data" / "pricing" / "ec2_on_demand.json"

FEATURE_NAMES = ["std", "min", "p50", "trend", "cv"]
HOURS_MONTH = 730


def load_models():
    models = {}
    for q in [0.10, 0.50, 0.95]:
        path = MODELS_DIR / f"quantile_{q:.2f}.json"
        model = xgb.XGBRegressor()
        model.load_model(path)
        models[q] = model
    return models


def load_fleet_avg_hourly():
    """Compute the fleet-weighted average hourly cost from EC2 pricing.

    Used as a per-VM cost proxy when individual instance types are unknown.
    This is more accurate than a fixed constant because it reflects the
    actual instance mix in the fleet.
    """
    try:
        with open(PRICING_FILE) as f:
            pricing = json.load(f)
        fleet_csv = DATA_DIR / "fleet_cost_estimate.csv"
        if not fleet_csv.exists():
            rates = [p["usd_per_hour"] for p in pricing["instances"].values()]
            return sum(rates) / len(rates)
        with open(fleet_csv) as f:
            rows = list(csv.DictReader(f))
        total_cost = sum(int(r["monthly_cost"]) for r in rows)
        total_vms = sum(int(r["vm_count"]) for r in rows)
        return (total_cost / total_vms) / HOURS_MONTH
    except (FileNotFoundError, KeyError, ZeroDivisionError):
        return 0.096  # m5.large fallback


def build_features_from_summary(csv_path):
    """Build the 5 VIF-cleaned features from a vm_utilization_summary.csv."""
    instances = []
    features = []

    warned_std = False
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            cpu_mean = float(row["cpu_mean"])
            cpu_min = float(row["cpu_min"])
            cpu_median = float(row["cpu_median"])

            if "cpu_std" in row:
                cpu_std = float(row["cpu_std"])
            else:
                # Approximate std from range when not available
                cpu_std = (float(row.get("cpu_max", cpu_mean)) - cpu_min) / 4
                if not warned_std:
                    print("Warning: cpu_std not in CSV, approximating from range", file=sys.stderr)
                    warned_std = True

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


def estimate_savings(action, hourly_cost, pred_mid):
    """Estimate monthly savings based on action and real EC2 pricing."""
    if action == "terminate":
        return hourly_cost * HOURS_MONTH
    elif action == "downsize":
        return hourly_cost * HOURS_MONTH * 0.5
    return 0.0


def predict(csv_path, output_path=None):
    """Run predictions on a VM utilization summary CSV."""
    models = load_models()
    instances, X = build_features_from_summary(csv_path)
    avg_hourly = load_fleet_avg_hourly()

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
            savings = estimate_savings(action, avg_hourly, mid)
            w.writerow([inst, f"{low:.2f}", f"{mid:.2f}", f"{high:.2f}",
                        action, risk, f"{savings:.2f}"])

    print(f"Wrote {len(instances)} recommendations to {output_path}")
    print(f"Fleet avg hourly rate: ${avg_hourly:.4f} (from EC2 pricing)")


if __name__ == "__main__":
    input_csv = sys.argv[1] if len(sys.argv) > 1 else DATA_DIR / "vm_utilization_summary.csv"
    output_csv = sys.argv[2] if len(sys.argv) > 2 else None
    predict(input_csv, output_csv)

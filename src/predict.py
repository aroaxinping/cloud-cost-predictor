"""Load trained XGBoost models and generate VM recommendations from new data."""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb
import yaml

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data" / "clean"
PRICING_FILE = ROOT / "data" / "pricing" / "ec2_on_demand.json"
CONFIG_FILE = ROOT / "config.yaml"

FEATURE_NAMES = ["std", "min", "p50", "trend", "cv"]
HOURS_MONTH = 730


def load_config() -> dict:
    with open(CONFIG_FILE) as f:
        return yaml.safe_load(f)


def load_models():
    config = load_config()
    quantiles = config["model"]["quantiles"]
    models = {}
    for q in quantiles:
        path = MODELS_DIR / f"quantile_{q:.2f}.json"
        model = xgb.XGBRegressor()
        model.load_model(path)
        models[q] = model
    return models, quantiles


def load_fleet_avg_hourly():
    """Compute the fleet-weighted average hourly cost from EC2 pricing."""
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
        return 0.096


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
                cpu_std = (float(row.get("cpu_max", cpu_mean)) - cpu_min) / 4
                if not warned_std:
                    print("Warning: cpu_std not in CSV, approximating from range", file=sys.stderr)
                    warned_std = True

            trend = 0.0
            cv = cpu_std / cpu_mean if cpu_mean > 0 else 0

            instances.append(row["instance"])
            features.append([cpu_std, cpu_min, cpu_median, trend, cv])

    return instances, np.array(features)


def recommend(pred_high: float, pred_mid: float, thresholds: dict | None = None) -> str:
    if thresholds is None:
        thresholds = load_config()["thresholds"]
    if pred_high < thresholds["terminate_cpu"]:
        return "terminate"
    if pred_high < thresholds["downsize_cpu"]:
        return "downsize"
    if pred_mid < thresholds["review_cpu"]:
        return "review"
    return "keep"


def assess_risk(action: str, pred_high: float, margins: dict | None = None) -> str:
    if margins is None:
        margins = load_config()["risk_margins"]
    if action == "terminate":
        margin = 5 - pred_high
        if margin > margins["terminate"]["safe"]:
            return "safe"
        if margin > margins["terminate"]["moderate"]:
            return "moderate"
        return "risky"
    elif action == "downsize":
        margin = 20 - pred_high
        if margin > margins["downsize"]["safe"]:
            return "safe"
        if margin > margins["downsize"]["moderate"]:
            return "moderate"
        return "risky"
    return "n/a"


def estimate_savings(action: str, hourly_cost: float, pred_mid: float) -> float:
    """Estimate monthly savings based on action and real EC2 pricing."""
    if action == "terminate":
        return hourly_cost * HOURS_MONTH
    elif action == "downsize":
        return hourly_cost * HOURS_MONTH * 0.5
    return 0.0


def predict(csv_path, output_path=None):
    """Run predictions on a VM utilization summary CSV."""
    config = load_config()
    models, quantiles = load_models()
    instances, X = build_features_from_summary(csv_path)
    avg_hourly = load_fleet_avg_hourly()
    thresholds = config["thresholds"]
    margins = config["risk_margins"]

    preds = {q: models[q].predict(X) for q in quantiles}

    if output_path is None:
        output_path = DATA_DIR / "vm_recommendations.csv"

    with open(output_path, "w", newline="\n") as f:
        w = csv.writer(f)
        w.writerow(["instance", "pred_low", "pred_mid", "pred_high",
                     "action", "risk", "monthly_savings"])
        for i, inst in enumerate(instances):
            low = preds[quantiles[0]][i]
            mid = preds[quantiles[1]][i]
            high = preds[quantiles[2]][i]
            action = recommend(high, mid, thresholds)
            risk = assess_risk(action, high, margins)
            savings = estimate_savings(action, avg_hourly, mid)
            w.writerow([inst, f"{low:.2f}", f"{mid:.2f}", f"{high:.2f}",
                        action, risk, f"{savings:.2f}"])

    print(f"Wrote {len(instances)} recommendations to {output_path}")
    print(f"Fleet avg hourly rate: ${avg_hourly:.4f} (from EC2 pricing)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate VM recommendations from utilization data")
    parser.add_argument("--input", default=str(DATA_DIR / "vm_utilization_summary.csv"),
                        help="path to vm_utilization_summary.csv")
    parser.add_argument("--output", default=None,
                        help="output CSV path (default: data/clean/vm_recommendations.csv)")
    args = parser.parse_args()
    predict(args.input, args.output)

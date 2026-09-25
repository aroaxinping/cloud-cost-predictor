"""Load trained XGBoost models and generate VM recommendations from new data."""
import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb
import yaml

from src.anomaly import detect_spikes
from src.memory_model import predict_memory

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data" / "clean"
PRICING_FILE = ROOT / "data" / "pricing" / "ec2_on_demand.json"
CONFIG_FILE = ROOT / "config.yaml"
DAILY_CPU_FILE = DATA_DIR / "vm_cpu_daily.csv"

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


ACTION_RANK = {"terminate": 0, "downsize": 1, "review": 2, "keep": 3}


def recommend_cpu(pred_high: float, pred_mid: float, thresholds: dict) -> str:
    if pred_high < thresholds["terminate_cpu"]:
        return "terminate"
    if pred_high < thresholds["downsize_cpu"]:
        return "downsize"
    if pred_mid < thresholds["review_cpu"]:
        return "review"
    return "keep"


def recommend_mem(mem_high: float, mem_mid: float, thresholds: dict) -> str:
    if mem_high < thresholds.get("terminate_mem", 10):
        return "terminate"
    if mem_high < thresholds.get("downsize_mem", 30):
        return "downsize"
    if mem_mid < thresholds.get("review_mem", 60):
        return "review"
    return "keep"


def recommend(pred_high: float, pred_mid: float, thresholds: dict | None = None,
              mem_high: float | None = None, mem_mid: float | None = None) -> str:
    t: dict = thresholds if thresholds is not None else load_config()["thresholds"]
    cpu_action = recommend_cpu(pred_high, pred_mid, t)
    if mem_high is not None and mem_mid is not None:
        mem_action = recommend_mem(mem_high, mem_mid, t)
        if ACTION_RANK[mem_action] > ACTION_RANK[cpu_action]:
            return mem_action
    return cpu_action


def assess_risk(action: str, pred_high: float, margins: dict | None = None,
                thresholds: dict | None = None) -> str:
    m: dict = margins if margins is not None else load_config()["risk_margins"]
    t: dict = thresholds if thresholds is not None else load_config()["thresholds"]
    if action == "terminate":
        margin = t["terminate_cpu"] - pred_high
        if margin > m["terminate"]["safe"]:
            return "safe"
        if margin > m["terminate"]["moderate"]:
            return "moderate"
        return "risky"
    elif action == "downsize":
        margin = t["downsize_cpu"] - pred_high
        if margin > m["downsize"]["safe"]:
            return "safe"
        if margin > m["downsize"]["moderate"]:
            return "moderate"
        return "risky"
    return "n/a"


def estimate_savings(action: str, hourly_cost: float) -> float:
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

    anomalies: dict[str, dict] = {}
    if DAILY_CPU_FILE.exists():
        anomaly_config = config.get("anomaly", {})
        anomalies = detect_spikes(DAILY_CPU_FILE, anomaly_config)
        n_spikes = sum(1 for a in anomalies.values() if a["spike"])
        logger.info("Anomaly detection: %d VMs flagged with recent spikes", n_spikes)

    actual_cpu = {}
    actual_mem = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            actual_cpu[row["instance"]] = row["cpu_mean"]
            actual_mem[row["instance"]] = row.get("mem_mean", "")

    preds = {q: models[q].predict(X) for q in quantiles}

    mem_preds = predict_memory(csv_path)
    has_mem = len(mem_preds) > 0
    if has_mem:
        logger.info("Memory model loaded: %d VMs with memory predictions", len(mem_preds))
    else:
        logger.info("No memory models found — using CPU-only recommendations")

    if output_path is None:
        output_path = DATA_DIR / "vm_recommendations.csv"

    overridden = 0
    mem_overridden = 0
    with open(output_path, "w", newline="\n") as f:
        w = csv.writer(f)
        w.writerow(["instance", "actual_cpu", "pred_low", "pred_mid", "pred_high",
                     "actual_mem", "mem_pred_low", "mem_pred_mid", "mem_pred_high",
                     "action", "risk", "monthly_savings", "spike_flag"])
        for i, inst in enumerate(instances):
            low = preds[quantiles[0]][i]
            mid = preds[quantiles[1]][i]
            high = preds[quantiles[2]][i]

            mp = mem_preds.get(inst, {})
            mem_low = mp.get("mem_low")
            mem_mid = mp.get("mem_mid")
            mem_high = mp.get("mem_high")

            cpu_action = recommend_cpu(high, mid, thresholds)
            action = recommend(high, mid, thresholds, mem_high, mem_mid)
            if action != cpu_action:
                mem_overridden += 1

            risk = assess_risk(action, high, margins, thresholds)
            savings = estimate_savings(action, avg_hourly)

            spike = anomalies.get(inst, {}).get("spike", False)
            if spike and action == "terminate":
                action = "review"
                risk = "spike"
                savings = 0.0
                overridden += 1

            w.writerow([
                inst, actual_cpu.get(inst, ""), f"{low:.2f}", f"{mid:.2f}", f"{high:.2f}",
                actual_mem.get(inst, ""),
                f"{mem_low:.2f}" if mem_low is not None else "",
                f"{mem_mid:.2f}" if mem_mid is not None else "",
                f"{mem_high:.2f}" if mem_high is not None else "",
                action, risk, f"{savings:.2f}", spike,
            ])

    print(f"Wrote {len(instances)} recommendations to {output_path}")
    print(f"Fleet avg hourly rate: ${avg_hourly:.4f} (from EC2 pricing)")
    if has_mem:
        print(f"Memory-aware: {mem_overridden} VMs escalated due to memory predictions")
    if overridden:
        print(f"Anomaly override: {overridden} VMs changed from terminate to review (recent spikes)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate VM recommendations from utilization data")
    parser.add_argument("--input", default=str(DATA_DIR / "vm_utilization_summary.csv"),
                        help="path to vm_utilization_summary.csv")
    parser.add_argument("--output", default=None,
                        help="output CSV path (default: data/clean/vm_recommendations.csv)")
    args = parser.parse_args()
    predict(args.input, args.output)

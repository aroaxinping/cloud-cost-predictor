"""Anomaly detection: flag VMs with recent CPU spikes.

A VM with a low monthly average but a recent spike should not be terminated —
the spike signals intermittent but real usage that the aggregate statistics miss.
This module scans the daily CPU time series and flags VMs whose recent peak
exceeds a z-score threshold relative to their own historical baseline.

When integrated into the prediction pipeline, flagged VMs have their action
overridden from "terminate" to "review" with a "spike" risk tag.
"""
import csv
import logging
from pathlib import Path

import numpy as np
import yaml

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "clean"
CONFIG_FILE = ROOT / "config.yaml"
DAILY_CPU_FILE = DATA_DIR / "vm_cpu_daily.csv"


def load_anomaly_config(config_path: Path | None = None) -> dict:
    config_path = config_path or CONFIG_FILE
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("anomaly", {
        "recent_days": 7,
        "z_threshold": 2.5,
        "min_observations": 5,
    })


def detect_spikes(
    daily_csv: Path | None = None,
    config: dict | None = None,
) -> dict[str, dict]:
    """Scan daily CPU time series for recent spikes.

    Returns {instance: {spike: bool, recent_max: float, z_score: float,
             historical_mean: float, historical_std: float}}.
    """
    daily_csv = daily_csv or DAILY_CPU_FILE
    if config is None:
        config = load_anomaly_config()

    recent_days = config.get("recent_days", 7)
    z_threshold = config.get("z_threshold", 2.5)
    min_obs = config.get("min_observations", 5)

    results: dict[str, dict] = {}

    with open(daily_csv) as f:
        reader = csv.DictReader(f)
        dates = [c for c in reader.fieldnames if c != "instance"]
        n_dates = len(dates)
        split = max(0, n_dates - recent_days)
        historical_dates = dates[:split]
        recent_dates = dates[split:]

        for row in reader:
            instance = row["instance"]

            historical_vals = []
            for d in historical_dates:
                v = row[d]
                if v:
                    try:
                        historical_vals.append(float(v))
                    except ValueError:
                        continue

            recent_vals = []
            for d in recent_dates:
                v = row[d]
                if v:
                    try:
                        recent_vals.append(float(v))
                    except ValueError:
                        continue

            if len(historical_vals) < min_obs or not recent_vals:
                results[instance] = {
                    "spike": False,
                    "recent_max": max(recent_vals) if recent_vals else 0.0,
                    "z_score": 0.0,
                    "historical_mean": np.mean(historical_vals) if historical_vals else 0.0,
                    "historical_std": 0.0,
                }
                continue

            hist_mean = np.mean(historical_vals)
            hist_std = np.std(historical_vals)
            recent_max = max(recent_vals)

            if hist_std > 0:
                z = (recent_max - hist_mean) / hist_std
            else:
                z = 0.0 if recent_max <= hist_mean else float("inf")

            results[instance] = {
                "spike": z >= z_threshold,
                "recent_max": round(recent_max, 2),
                "z_score": round(z, 2),
                "historical_mean": round(hist_mean, 2),
                "historical_std": round(hist_std, 2),
            }

    n_spikes = sum(1 for r in results.values() if r["spike"])
    logger.info("Anomaly detection: %d/%d VMs flagged (z >= %.1f over last %d days)",
                n_spikes, len(results), z_threshold, recent_days)
    return results


def write_anomaly_report(results: dict[str, dict], output_path: Path | None = None) -> Path:
    output_path = output_path or (DATA_DIR / "vm_anomalies.csv")
    flagged = {k: v for k, v in results.items() if v["spike"]}

    with open(output_path, "w", newline="\n") as f:
        w = csv.writer(f)
        w.writerow(["instance", "recent_max", "z_score", "historical_mean", "historical_std"])
        for inst in sorted(flagged, key=lambda k: -flagged[k]["z_score"]):
            r = flagged[inst]
            w.writerow([inst, r["recent_max"], r["z_score"],
                        r["historical_mean"], r["historical_std"]])

    logger.info("Wrote %d anomalies to %s", len(flagged), output_path)
    return output_path


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Detect CPU spike anomalies in VM fleet")
    parser.add_argument("--input", default=str(DAILY_CPU_FILE),
                        help="path to vm_cpu_daily.csv")
    parser.add_argument("--output", default=None,
                        help="output CSV path (default: data/clean/vm_anomalies.csv)")
    args = parser.parse_args()

    results = detect_spikes(Path(args.input))
    out = write_anomaly_report(results, Path(args.output) if args.output else None)

    n_spikes = sum(1 for r in results.values() if r["spike"])
    print(f"Flagged {n_spikes:,} of {len(results):,} VMs with recent spikes")
    print(f"Report: {out}")

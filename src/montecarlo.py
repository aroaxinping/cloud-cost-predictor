"""Monte Carlo simulation to propagate prediction uncertainty into savings estimates."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE = ROOT / "config.yaml"
HOURS_MONTH = 730


def load_config() -> dict:
    with open(CONFIG_FILE) as f:
        return yaml.safe_load(f)


def simulate_savings(
    pred_low: np.ndarray,
    pred_mid: np.ndarray,
    pred_high: np.ndarray,
    hourly_cost: float,
    n_simulations: int = 10_000,
    rng_seed: int = 42,
) -> dict:
    """Run Monte Carlo simulation on fleet savings.

    For each VM, sample future CPU from a distribution bounded by the quantile
    predictions (triangular: min=pred_low, mode=pred_mid, max=pred_high).
    Then classify each sample and compute savings. Returns distribution stats.
    """
    config = load_config()
    t = config["thresholds"]
    rng = np.random.default_rng(rng_seed)
    n_vms = len(pred_mid)

    low = np.clip(pred_low, 0, 100)
    mid = np.clip(pred_mid, low, 100)
    high = np.clip(pred_high, mid, 100)
    # Triangular needs low < high; handle degenerate cases
    spread = high - low
    spread = np.where(spread < 0.01, 0.01, spread)
    high_adj = low + spread

    monthly_savings = np.zeros(n_simulations)

    for sim in range(n_simulations):
        sampled_cpu = rng.triangular(low, mid, high_adj)

        terminate_mask = sampled_cpu < t["terminate_cpu"]
        downsize_mask = (~terminate_mask) & (sampled_cpu < t["downsize_cpu"])

        savings = np.zeros(n_vms)
        savings[terminate_mask] = hourly_cost * HOURS_MONTH
        savings[downsize_mask] = hourly_cost * HOURS_MONTH * 0.5

        monthly_savings[sim] = savings.sum()

    return {
        "mean": float(np.mean(monthly_savings)),
        "median": float(np.median(monthly_savings)),
        "std": float(np.std(monthly_savings)),
        "p5": float(np.percentile(monthly_savings, 5)),
        "p25": float(np.percentile(monthly_savings, 25)),
        "p75": float(np.percentile(monthly_savings, 75)),
        "p95": float(np.percentile(monthly_savings, 95)),
        "n_simulations": n_simulations,
        "n_vms": n_vms,
    }

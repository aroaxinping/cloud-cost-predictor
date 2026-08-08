"""Tests for Monte Carlo savings simulation."""
import numpy as np

from src.montecarlo import simulate_savings


def test_all_zero_cpu_means_full_termination():
    n = 100
    low = np.zeros(n)
    mid = np.ones(n) * 0.5
    high = np.ones(n) * 1.0
    result = simulate_savings(low, mid, high, hourly_cost=0.10, n_simulations=500)
    assert result["mean"] > 0
    assert result["n_vms"] == n
    assert result["p5"] <= result["median"] <= result["p95"]


def test_high_cpu_means_no_savings():
    n = 50
    low = np.ones(n) * 80
    mid = np.ones(n) * 90
    high = np.ones(n) * 95
    result = simulate_savings(low, mid, high, hourly_cost=0.10, n_simulations=500)
    assert result["mean"] == 0.0


def test_deterministic_with_seed():
    n = 200
    low = np.random.default_rng(0).uniform(0, 5, n)
    mid = low + 2
    high = mid + 5
    r1 = simulate_savings(low, mid, high, hourly_cost=0.10, rng_seed=99)
    r2 = simulate_savings(low, mid, high, hourly_cost=0.10, rng_seed=99)
    assert r1["mean"] == r2["mean"]

"""Tests for src/anomaly.py — CPU spike detection."""
import csv
import textwrap
from pathlib import Path

import pytest

from src.anomaly import detect_spikes, load_anomaly_config, write_anomaly_report


@pytest.fixture
def daily_csv(tmp_path):
    """Create a minimal daily CPU CSV with a known spike."""
    path = tmp_path / "vm_cpu_daily.csv"
    with open(path, "w", newline="\n") as f:
        w = csv.writer(f)
        dates = [f"2024-08-{d:02d}" for d in range(1, 32)]
        w.writerow(["instance"] + dates)
        # Stable VM: all values around 3%
        stable = ["3.0"] * 31
        w.writerow(["vm-stable"] + stable)
        # Spike VM: 24 days at ~2%, then spikes to 80% on day 28
        spike = ["2.0"] * 24 + ["2.0", "2.0", "2.0", "80.0", "2.0", "2.0", "2.0"]
        w.writerow(["vm-spike"] + spike)
        # Consistently high VM: always ~60%
        high = ["60.0"] * 31
        w.writerow(["vm-high"] + high)
        # Sparse VM: very few observations
        sparse = [""] * 28 + ["5.0", "", "5.0"]
        w.writerow(["vm-sparse"] + sparse)
    return path


@pytest.fixture
def config():
    return {"recent_days": 7, "z_threshold": 2.5, "min_observations": 5}


def test_detect_spikes_flags_spike_vm(daily_csv, config):
    results = detect_spikes(daily_csv, config)
    assert results["vm-spike"]["spike"] is True
    assert results["vm-spike"]["recent_max"] == 80.0
    assert results["vm-spike"]["z_score"] > 2.5


def test_detect_spikes_stable_not_flagged(daily_csv, config):
    results = detect_spikes(daily_csv, config)
    assert results["vm-stable"]["spike"] is False


def test_detect_spikes_high_not_flagged(daily_csv, config):
    results = detect_spikes(daily_csv, config)
    assert results["vm-high"]["spike"] is False


def test_detect_spikes_sparse_not_flagged(daily_csv, config):
    results = detect_spikes(daily_csv, config)
    assert results["vm-sparse"]["spike"] is False


def test_detect_spikes_returns_all_vms(daily_csv, config):
    results = detect_spikes(daily_csv, config)
    assert len(results) == 4


def test_detect_spikes_stricter_threshold(daily_csv):
    """With z=100 threshold, only vm-spike (inf z due to zero std) is still flagged."""
    config = {"recent_days": 7, "z_threshold": 100.0, "min_observations": 5}
    results = detect_spikes(daily_csv, config)
    flagged = [k for k, v in results.items() if v["spike"]]
    assert flagged == ["vm-spike"]


def test_detect_spikes_looser_threshold(daily_csv):
    config = {"recent_days": 7, "z_threshold": 0.1, "min_observations": 5}
    results = detect_spikes(daily_csv, config)
    flagged = [k for k, v in results.items() if v["spike"]]
    assert "vm-spike" in flagged


def test_write_anomaly_report(daily_csv, config, tmp_path):
    results = detect_spikes(daily_csv, config)
    out = write_anomaly_report(results, tmp_path / "anomalies.csv")
    assert out.exists()
    with open(out) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) >= 1
    assert rows[0]["instance"] == "vm-spike"
    assert float(rows[0]["z_score"]) > 2.5


def test_write_anomaly_report_empty(tmp_path):
    results = {"vm-ok": {"spike": False, "recent_max": 3.0, "z_score": 0.5,
                          "historical_mean": 3.0, "historical_std": 0.5}}
    out = write_anomaly_report(results, tmp_path / "anomalies.csv")
    with open(out) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 0


def test_load_anomaly_config():
    config = load_anomaly_config()
    assert "recent_days" in config
    assert "z_threshold" in config
    assert "min_observations" in config
    assert config["recent_days"] == 7
    assert config["z_threshold"] == 2.5


def test_detect_spikes_custom_recent_days(daily_csv):
    config = {"recent_days": 3, "z_threshold": 2.5, "min_observations": 5}
    results = detect_spikes(daily_csv, config)
    assert "vm-spike" in results


def test_zero_std_vm(tmp_path):
    """A VM with zero variance should not crash and should not flag unless recent > mean."""
    path = tmp_path / "zero_std.csv"
    with open(path, "w", newline="\n") as f:
        w = csv.writer(f)
        dates = [f"2024-08-{d:02d}" for d in range(1, 32)]
        w.writerow(["instance"] + dates)
        w.writerow(["vm-flat"] + ["5.0"] * 31)
    config = {"recent_days": 7, "z_threshold": 2.5, "min_observations": 5}
    results = detect_spikes(path, config)
    assert results["vm-flat"]["spike"] is False
    assert results["vm-flat"]["z_score"] == 0.0

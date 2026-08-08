"""Tests for memory quantile prediction model."""
import csv
import tempfile

import numpy as np

from src.memory_model import build_memory_features, MEM_FEATURE_NAMES


def _write_summary_csv(rows, include_mem_std=False):
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    header = ["instance", "mem_mean", "mem_median", "mem_p5", "mem_p95",
              "mem_min", "mem_max", "mem_n"]
    if include_mem_std:
        header.append("mem_std")
    w = csv.writer(f)
    w.writerow(header)
    for row in rows:
        w.writerow(row)
    f.flush()
    return f.name


def test_build_features_basic():
    path = _write_summary_csv([
        ["vm-1", "50.0", "48.0", "10.0", "90.0", "5.0", "95.0", "100"],
    ])
    instances, X = build_memory_features(path)
    assert instances == ["vm-1"]
    assert X.shape == (1, 5)
    assert X[0, 1] == 5.0   # mem_min
    assert X[0, 2] == 48.0  # mem_p50 (median)
    assert X[0, 3] == 0.0   # mem_trend


def test_build_features_std_approximation():
    path = _write_summary_csv([
        ["vm-1", "50.0", "48.0", "10.0", "90.0", "5.0", "95.0", "100"],
    ])
    instances, X = build_memory_features(path)
    expected_std = (95.0 - 5.0) / 4
    assert abs(X[0, 0] - expected_std) < 0.01


def test_build_features_with_mem_std():
    path = _write_summary_csv([
        ["vm-1", "50.0", "48.0", "10.0", "90.0", "5.0", "95.0", "100", "12.5"],
    ], include_mem_std=True)
    instances, X = build_memory_features(path)
    assert abs(X[0, 0] - 12.5) < 0.01


def test_build_features_skips_zero_n():
    path = _write_summary_csv([
        ["vm-1", "50.0", "48.0", "10.0", "90.0", "5.0", "95.0", "0"],
        ["vm-2", "30.0", "28.0", "5.0", "60.0", "2.0", "70.0", "50"],
    ])
    instances, X = build_memory_features(path)
    assert instances == ["vm-2"]
    assert X.shape == (1, 5)


def test_build_features_cv():
    path = _write_summary_csv([
        ["vm-1", "50.0", "48.0", "10.0", "90.0", "10.0", "90.0", "100"],
    ])
    _, X = build_memory_features(path)
    mem_std = (90.0 - 10.0) / 4
    expected_cv = mem_std / 50.0
    assert abs(X[0, 4] - expected_cv) < 0.01


def test_feature_names_match():
    assert MEM_FEATURE_NAMES == ["mem_std", "mem_min", "mem_p50", "mem_trend", "mem_cv"]

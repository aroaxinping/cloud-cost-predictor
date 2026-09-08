"""Tests for data validation module."""
import csv
import tempfile
from pathlib import Path

from src.validate import validate_summary


def _write_csv(rows: list[list[str]]) -> Path:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    w = csv.writer(f)
    for row in rows:
        w.writerow(row)
    f.flush()
    return Path(f.name)


HEADER = [
    "instance", "cpu_mean", "cpu_std", "cpu_median", "cpu_p5", "cpu_p95",
    "cpu_min", "cpu_max", "cpu_n", "mem_mean", "mem_std", "mem_median",
    "mem_p5", "mem_p95", "mem_min", "mem_max", "mem_n",
]
VALID_ROW = ["vm-1", "25.0", "5.0", "24.0", "10.0", "50.0", "5.0", "60.0", "100",
             "40.0", "8.0", "38.0", "20.0", "70.0", "15.0", "80.0", "100"]


def test_valid_file_passes():
    path = _write_csv([HEADER, VALID_ROW])
    assert validate_summary(path) == []


def test_missing_column():
    path = _write_csv([["instance", "cpu_mean"]])
    errors = validate_summary(path)
    assert any("Missing columns" in e for e in errors)


def test_empty_file():
    path = _write_csv([HEADER])
    errors = validate_summary(path)
    assert any("empty" in e.lower() for e in errors)


def test_duplicate_instance():
    path = _write_csv([HEADER, VALID_ROW, VALID_ROW])
    errors = validate_summary(path)
    assert any("duplicate" in e for e in errors)


def test_negative_value():
    bad_row = VALID_ROW.copy()
    bad_row[1] = "-5.0"
    path = _write_csv([HEADER, bad_row])
    errors = validate_summary(path)
    assert any("negative" in e for e in errors)


def test_cpu_over_100():
    bad_row = VALID_ROW.copy()
    bad_row[1] = "150.0"
    path = _write_csv([HEADER, bad_row])
    errors = validate_summary(path)
    assert any("exceeds 100%" in e for e in errors)


def test_file_not_found():
    errors = validate_summary(Path("/nonexistent/file.csv"))
    assert any("not found" in e.lower() for e in errors)

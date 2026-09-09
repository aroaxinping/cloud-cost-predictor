"""Integration tests: end-to-end pipeline validation."""
from pathlib import Path

import pytest

from src.validate import validate_summary


def test_real_data_passes_validation():
    """The actual clean data file should pass all validation checks."""
    path = Path("data/clean/vm_utilization_summary.csv")
    if not path.exists():
        pytest.skip("Clean data not available (run make ingest first)")
    errors = validate_summary(path)
    assert errors == [], f"Real data has {len(errors)} validation errors: {errors[:5]}"


def test_sample_fixture_passes_validation(sample_summary_csv: Path):
    """The test fixture should also pass validation."""
    errors = validate_summary(sample_summary_csv)
    assert errors == []

"""Tests for the ingest module."""
import csv
import zipfile
from pathlib import Path

import pytest

from src.ingest import build, summarize_metric


@pytest.fixture
def fake_zip(tmp_path: Path) -> Path:
    """Create a minimal fake SAP zip with CPU and memory CSVs."""
    zip_path = tmp_path / "sap.zip"
    prefix = "sap-cloud-infrastructure-dataset/data/"

    cpu_csv = "Instance,Value\nvm-001,10.0\nvm-001,20.0\nvm-001,30.0\nvm-002,50.0\nvm-002,60.0\n"
    mem_csv = "Instance,Value\nvm-001,40.0\nvm-001,45.0\nvm-001,50.0\nvm-002,70.0\nvm-002,80.0\n"
    size_csv = "RAM_Category,VCPU_Category,Count\nSmall,Small,100\nMedium,Medium,200\n"

    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(prefix + "vrops_virtualmachine_cpu_usage_ratio_all.csv", cpu_csv)
        zf.writestr(prefix + "vrops_virtualmachine_memory_usage_ratio_all.csv", mem_csv)
        zf.writestr(prefix + "vm_classification_30d.csv", size_csv)

    return zip_path


def test_summarize_metric_computes_stats(fake_zip: Path):
    with zipfile.ZipFile(fake_zip) as zf:
        result = summarize_metric(zf, "vrops_virtualmachine_cpu_usage_ratio_all.csv")

    assert "vm-001" in result
    assert "vm-002" in result
    stats = result["vm-001"]
    assert stats["n"] == 3
    assert abs(stats["mean"] - 20.0) < 0.01
    assert stats["min"] == 10.0
    assert stats["max"] == 30.0


def test_summarize_metric_handles_two_instances(fake_zip: Path):
    with zipfile.ZipFile(fake_zip) as zf:
        result = summarize_metric(zf, "vrops_virtualmachine_cpu_usage_ratio_all.csv")

    assert len(result) == 2
    assert result["vm-002"]["n"] == 2


def test_build_produces_output_files(fake_zip: Path, tmp_path: Path, monkeypatch):
    monkeypatch.setattr("src.ingest.ZIP", fake_zip)
    out_dir = tmp_path / "output"
    out_dir.mkdir()

    summary_path, sizes_path = build(out_dir)

    assert summary_path.exists()
    assert sizes_path.exists()

    with open(summary_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 2
    assert rows[0]["instance"] == "vm-001"
    assert float(rows[0]["cpu_mean"]) == pytest.approx(20.0, abs=0.01)


def test_build_missing_zip_raises(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("src.ingest.ZIP", tmp_path / "nonexistent.zip")
    with pytest.raises(FileNotFoundError):
        build(tmp_path / "output")

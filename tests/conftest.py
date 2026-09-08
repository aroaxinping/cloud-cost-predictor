"""Shared test fixtures."""
import csv
import tempfile
from pathlib import Path

import pytest


SUMMARY_HEADER = [
    "instance", "cpu_mean", "cpu_median", "cpu_p5", "cpu_p95",
    "cpu_min", "cpu_max", "cpu_n", "mem_mean", "mem_median",
    "mem_p5", "mem_p95", "mem_min", "mem_max", "mem_n",
]


def _make_vm_row(
    instance: str = "vm-1",
    cpu_mean: float = 25.0,
    cpu_std: float = 5.0,
    mem_mean: float = 40.0,
) -> list[str]:
    cpu_min = max(0.0, cpu_mean * 0.2)
    cpu_max = min(100.0, cpu_mean * 1.5 + 10)
    mem_min = max(0.0, mem_mean * 0.3)
    mem_max = min(100.0, mem_mean * 1.4 + 10)
    return [
        instance,
        str(cpu_mean), str(cpu_mean), str(max(0, cpu_mean * 0.5)),
        str(min(100, cpu_mean * 1.3)), str(cpu_min), str(cpu_max), "100",
        str(mem_mean), str(mem_mean), str(max(0, mem_mean * 0.5)),
        str(min(100, mem_mean * 1.3)), str(mem_min), str(mem_max), "100",
    ]


@pytest.fixture
def sample_summary_csv(tmp_path: Path) -> Path:
    """Create a minimal valid vm_utilization_summary.csv."""
    path = tmp_path / "vm_utilization_summary.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(SUMMARY_HEADER)
        w.writerow(_make_vm_row("zombie-1", cpu_mean=2.0, cpu_std=0.5, mem_mean=10.0))
        w.writerow(_make_vm_row("idle-1", cpu_mean=6.0, cpu_std=1.0, mem_mean=25.0))
        w.writerow(_make_vm_row("oversized-1", cpu_mean=15.0, cpu_std=3.0, mem_mean=30.0))
        w.writerow(_make_vm_row("right-1", cpu_mean=50.0, cpu_std=10.0, mem_mean=60.0))
        w.writerow(_make_vm_row("hot-1", cpu_mean=90.0, cpu_std=3.0, mem_mean=80.0))
    return path

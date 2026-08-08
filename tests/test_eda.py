"""Tests for VM classification and fleet analysis."""
import csv

from src.eda import classify_vm, fleet_summary, load_and_classify, write_classified

THRESHOLDS = {
    "zombie_cpu_max": 5,
    "zombie_mem_max": 20,
    "idle_cpu_p95_max": 10,
    "idle_mem_max": 50,
    "oversized_cpu_mean_max": 20,
    "oversized_cpu_p95_max": 50,
    "oversized_mem_max": 50,
    "hot_cpu_min": 80,
    "memory_bound_min": 80,
}


def test_classify_zombie():
    assert classify_vm(cpu_mean=2.0, mem_mean=10.0, cpu_p95=3.0, thresholds=THRESHOLDS) == "zombie"


def test_classify_idle():
    assert classify_vm(cpu_mean=6.0, mem_mean=25.0, cpu_p95=8.0, thresholds=THRESHOLDS) == "idle"


def test_classify_oversized():
    assert classify_vm(cpu_mean=15.0, mem_mean=30.0, cpu_p95=40.0, thresholds=THRESHOLDS) == "oversized"


def test_classify_right_sized():
    assert classify_vm(cpu_mean=50.0, mem_mean=60.0, cpu_p95=70.0, thresholds=THRESHOLDS) == "right-sized"


def test_classify_hot():
    assert classify_vm(cpu_mean=90.0, mem_mean=80.0, cpu_p95=95.0, thresholds=THRESHOLDS) == "hot"


def test_classify_review():
    assert classify_vm(cpu_mean=15.0, mem_mean=30.0, cpu_p95=55.0, thresholds=THRESHOLDS) == "review"


def test_zombie_requires_both_low_cpu_and_mem():
    assert classify_vm(cpu_mean=3.0, mem_mean=50.0, cpu_p95=5.0, thresholds=THRESHOLDS) == "review"


def test_high_memory_prevents_idle():
    assert classify_vm(cpu_mean=3.0, mem_mean=85.0, cpu_p95=5.0, thresholds=THRESHOLDS) == "right-sized"


def test_high_memory_prevents_zombie():
    assert classify_vm(cpu_mean=2.0, mem_mean=90.0, cpu_p95=3.0, thresholds=THRESHOLDS) == "right-sized"


def test_moderate_memory_allows_idle():
    assert classify_vm(cpu_mean=4.0, mem_mean=40.0, cpu_p95=8.0, thresholds=THRESHOLDS) == "idle"


def test_fleet_summary_counts():
    vms = [
        {"class": "zombie"}, {"class": "zombie"},
        {"class": "idle"}, {"class": "idle"}, {"class": "idle"},
        {"class": "oversized"},
        {"class": "right-sized"},
        {"class": "hot"},
    ]
    result = fleet_summary(vms)
    assert result["total"] == 8
    assert result["classes"]["zombie"] == 2
    assert result["classes"]["idle"] == 3
    assert result["waste_count"] == 6  # zombie + idle + oversized


def test_fleet_summary_no_waste():
    vms = [{"class": "right-sized"}, {"class": "hot"}]
    result = fleet_summary(vms)
    assert result["waste_count"] == 0


def test_load_and_classify_from_csv(sample_summary_csv, monkeypatch):
    import src.eda
    monkeypatch.setattr(src.eda, "DATA", sample_summary_csv.parent)

    vms = load_and_classify(thresholds=THRESHOLDS)
    assert len(vms) == 5
    classes = {v["instance"]: v["class"] for v in vms}
    assert classes["zombie-1"] == "zombie"
    assert classes["hot-1"] == "hot"


def test_write_classified_creates_csv(tmp_path):
    vms = [
        {"instance": "vm-a", "class": "zombie", "cpu_mean": 2.0, "cpu_median": 1.8,
         "cpu_p5": 0.5, "cpu_p95": 3.0, "mem_mean": 10.0, "mem_median": 9.0,
         "mem_p5": 5.0, "mem_p95": 15.0},
        {"instance": "vm-b", "class": "hot", "cpu_mean": 90.0, "cpu_median": 88.0,
         "cpu_p5": 70.0, "cpu_p95": 95.0, "mem_mean": 80.0, "mem_median": 78.0,
         "mem_p5": 60.0, "mem_p95": 90.0},
    ]
    out = tmp_path / "classified.csv"
    write_classified(vms, out)
    assert out.exists()

    with open(out) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[0]["instance"] == "vm-a"  # sorted by cpu_mean
    assert rows[0]["class"] == "zombie"


def test_classify_with_custom_thresholds():
    strict = {**THRESHOLDS, "zombie_cpu_max": 1}
    assert classify_vm(cpu_mean=2.0, mem_mean=10.0, cpu_p95=3.0, thresholds=strict) != "zombie"

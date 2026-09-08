"""Tests for VM classification logic."""
from src.eda import classify_vm


def test_classify_zombie():
    assert classify_vm(cpu_mean=2.0, mem_mean=10.0, cpu_p95=3.0) == "zombie"


def test_classify_idle():
    assert classify_vm(cpu_mean=6.0, mem_mean=25.0, cpu_p95=8.0) == "idle"


def test_classify_oversized():
    assert classify_vm(cpu_mean=15.0, mem_mean=30.0, cpu_p95=40.0) == "oversized"


def test_classify_right_sized():
    assert classify_vm(cpu_mean=50.0, mem_mean=60.0, cpu_p95=70.0) == "right-sized"


def test_classify_hot():
    assert classify_vm(cpu_mean=90.0, mem_mean=80.0, cpu_p95=95.0) == "hot"


def test_classify_review():
    assert classify_vm(cpu_mean=15.0, mem_mean=30.0, cpu_p95=55.0) == "review"


def test_zombie_requires_both_low_cpu_and_mem():
    assert classify_vm(cpu_mean=3.0, mem_mean=50.0, cpu_p95=5.0) == "idle"

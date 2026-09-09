"""Tests for pricing module."""
import csv

from src.pricing import (
    DOWNSIZE_TO,
    HOURS_PER_MONTH,
    SIZE_TO_EC2,
    _load_classification_rates,
    get_price,
    load_ec2_prices,
)


def test_all_ec2_types_have_downsize_path():
    smallest = "t3.small"
    for ec2_type in SIZE_TO_EC2.values():
        if ec2_type != smallest:
            assert ec2_type in DOWNSIZE_TO, f"{ec2_type} has no downsize target"


def test_downsize_chain_terminates():
    for start in DOWNSIZE_TO:
        visited = set()
        current = start
        while current in DOWNSIZE_TO:
            assert current not in visited, f"Cycle detected at {current}"
            visited.add(current)
            current = DOWNSIZE_TO[current]


def test_size_mapping_covers_common_sizes():
    expected = {"t3.small", "t3.medium", "m5.large", "m5.xlarge", "r5.large"}
    actual = set(SIZE_TO_EC2.values())
    assert expected.issubset(actual)


def test_load_ec2_prices():
    prices = load_ec2_prices()
    assert "t3.small" in prices
    assert "r5.xlarge" in prices
    assert prices["t3.small"]["usd_per_hour"] > 0


def test_get_price():
    prices = load_ec2_prices()
    price = get_price(prices, "m5.large")
    assert price == 0.096


def test_downsize_always_cheaper():
    prices = load_ec2_prices()
    for big, small in DOWNSIZE_TO.items():
        if big in prices and small in prices:
            assert prices[big]["usd_per_hour"] > prices[small]["usd_per_hour"], (
                f"{big} should be more expensive than {small}"
            )


def test_load_classification_rates_returns_tuple():
    zombie, downsize = _load_classification_rates()
    assert 0 <= zombie <= 1
    assert 0 <= downsize <= 1


def test_load_classification_rates_fallback(tmp_path, monkeypatch):
    import src.pricing
    monkeypatch.setattr(src.pricing, "DATA", tmp_path)
    zombie, downsize = _load_classification_rates()
    assert zombie == 0.109
    assert downsize == 0.842


def test_load_classification_rates_from_csv(tmp_path, monkeypatch):
    import src.pricing
    monkeypatch.setattr(src.pricing, "DATA", tmp_path)

    classified = tmp_path / "vm_classified.csv"
    with open(classified, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["instance", "class"])
        for i in range(10):
            w.writerow([f"vm-{i}", "zombie"])
        for i in range(40):
            w.writerow([f"vm-idle-{i}", "idle"])
        for i in range(50):
            w.writerow([f"vm-ok-{i}", "right-sized"])

    zombie, downsize = _load_classification_rates()
    assert zombie == 0.1  # 10/100
    assert downsize == 0.4  # 40/100 (idle counts as downsize candidate)


def test_hours_per_month_is_730():
    assert HOURS_PER_MONTH == 730

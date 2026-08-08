"""Tests for Reserved Instance modeling."""
import csv
from pathlib import Path

import pytest

from src.reserved_instances import (
    HOURS_PER_MONTH,
    TERMS,
    build_ri_recommendations,
    compare_commitments,
    effective_hourly_rate,
    fleet_type_weights,
    fleet_weighted_on_demand_hourly,
    fleet_weighted_ri_hourly,
    load_keep_vms,
    write_ri_recommendations,
)

# A tiny, self-consistent fixture: two instance types, both terms, all
# payment options. Numbers are made up for the test but shaped like the
# real ec2_reserved_instances.json.
RI_PRICES = {
    "t3.small": {
        "1yr": {
            "No Upfront": {"hourly_usd": 0.0130, "upfront_usd": 0.0},
            "Partial Upfront": {"hourly_usd": 0.0062, "upfront_usd": 54.0},
            "All Upfront": {"hourly_usd": 0.0, "upfront_usd": 107.0},
        },
        "3yr": {
            "No Upfront": {"hourly_usd": 0.0090, "upfront_usd": 0.0},
            "Partial Upfront": {"hourly_usd": 0.0042, "upfront_usd": 109.0},
            "All Upfront": {"hourly_usd": 0.0, "upfront_usd": 206.0},
        },
    },
    "m5.large": {
        "1yr": {
            "No Upfront": {"hourly_usd": 0.0600, "upfront_usd": 0.0},
            "Partial Upfront": {"hourly_usd": 0.0290, "upfront_usd": 252.0},
            "All Upfront": {"hourly_usd": 0.0, "upfront_usd": 494.0},
        },
        "3yr": {
            "No Upfront": {"hourly_usd": 0.0410, "upfront_usd": 0.0},
            "Partial Upfront": {"hourly_usd": 0.0190, "upfront_usd": 505.0},
            "All Upfront": {"hourly_usd": 0.0, "upfront_usd": 949.0},
        },
    },
}

ON_DEMAND_PRICES = {
    "t3.small": {"vcpus": 2, "ram_gb": 2, "usd_per_hour": 0.0208},
    "m5.large": {"vcpus": 2, "ram_gb": 8, "usd_per_hour": 0.0960},
}

WEIGHTS = {"t3.small": 0.5, "m5.large": 0.5}


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


# --- effective_hourly_rate -------------------------------------------------


def test_effective_hourly_no_upfront_is_just_the_hourly_rate():
    entry = {"hourly_usd": 0.05, "upfront_usd": 0.0}
    assert effective_hourly_rate(entry, TERMS["1yr"]) == 0.05


def test_effective_hourly_all_upfront_amortizes_the_lump_sum():
    entry = {"hourly_usd": 0.0, "upfront_usd": 876.0}
    # 876 / 8760 hours in a 1yr term = 0.10 $/hr
    assert effective_hourly_rate(entry, 8760) == pytest.approx(0.10)


def test_effective_hourly_partial_upfront_combines_both():
    entry = {"hourly_usd": 0.02, "upfront_usd": 876.0}
    assert effective_hourly_rate(entry, 8760) == pytest.approx(0.12)


# --- fleet_type_weights -----------------------------------------------------


def test_fleet_type_weights_sum_to_one(tmp_path: Path):
    path = tmp_path / "vm_size_distribution.csv"
    _write_csv(
        path,
        ["ram_category", "vcpu_category", "total_count"],
        [
            ["Small", "Small", "300"],
            ["Medium", "Medium", "700"],
        ],
    )
    weights = fleet_type_weights(path)
    assert weights == {"t3.small": pytest.approx(0.3), "m5.large": pytest.approx(0.7)}


def test_fleet_type_weights_skips_unmapped_sizes(tmp_path: Path):
    path = tmp_path / "vm_size_distribution.csv"
    _write_csv(
        path,
        ["ram_category", "vcpu_category", "total_count"],
        [
            ["Small", "Small", "100"],
            ["Bogus", "Category", "900"],
        ],
    )
    weights = fleet_type_weights(path)
    assert set(weights) == {"t3.small"}


def test_fleet_type_weights_empty_file_raises(tmp_path: Path):
    path = tmp_path / "vm_size_distribution.csv"
    _write_csv(path, ["ram_category", "vcpu_category", "total_count"], [])
    with pytest.raises(ValueError):
        fleet_type_weights(path)


# --- weighted rates ----------------------------------------------------------


def test_fleet_weighted_on_demand_hourly_is_the_weighted_mean():
    hourly = fleet_weighted_on_demand_hourly(WEIGHTS, ON_DEMAND_PRICES)
    assert hourly == pytest.approx(0.5 * 0.0208 + 0.5 * 0.0960)


def test_fleet_weighted_ri_hourly_matches_manual_amortization():
    hourly = fleet_weighted_ri_hourly(WEIGHTS, "1yr", "Partial Upfront", RI_PRICES)
    t3_eff = effective_hourly_rate(RI_PRICES["t3.small"]["1yr"]["Partial Upfront"], TERMS["1yr"])
    m5_eff = effective_hourly_rate(RI_PRICES["m5.large"]["1yr"]["Partial Upfront"], TERMS["1yr"])
    assert hourly == pytest.approx(0.5 * t3_eff + 0.5 * m5_eff)


def test_fleet_weighted_hourly_raises_when_no_overlap():
    with pytest.raises(ValueError):
        fleet_weighted_ri_hourly({"r5.8xlarge": 1.0}, "1yr", "No Upfront", RI_PRICES)


# --- compare_commitments -----------------------------------------------------


def test_compare_commitments_covers_every_term_and_option():
    on_demand_hourly = fleet_weighted_on_demand_hourly(WEIGHTS, ON_DEMAND_PRICES)
    results = compare_commitments(WEIGHTS, RI_PRICES, on_demand_hourly)
    seen = {(r["term"], r["payment_option"]) for r in results}
    expected = {(t, o) for t in TERMS for o in ("No Upfront", "Partial Upfront", "All Upfront")}
    assert seen == expected


def test_compare_commitments_3yr_all_upfront_is_cheapest():
    """With realistic RI pricing, longer term + more upfront = lower effective rate."""
    on_demand_hourly = fleet_weighted_on_demand_hourly(WEIGHTS, ON_DEMAND_PRICES)
    results = compare_commitments(WEIGHTS, RI_PRICES, on_demand_hourly)
    cheapest = min(results, key=lambda r: r["effective_hourly"])
    assert (cheapest["term"], cheapest["payment_option"]) == ("3yr", "All Upfront")


def test_compare_commitments_savings_and_breakeven_are_complementary():
    on_demand_hourly = fleet_weighted_on_demand_hourly(WEIGHTS, ON_DEMAND_PRICES)
    results = compare_commitments(WEIGHTS, RI_PRICES, on_demand_hourly)
    for r in results:
        assert r["savings_pct"] + r["breakeven_utilization_pct"] == pytest.approx(100.0)


def test_compare_commitments_all_options_cheaper_than_on_demand():
    """Every RI option in this fixture should beat on-demand (sanity check on fixture)."""
    on_demand_hourly = fleet_weighted_on_demand_hourly(WEIGHTS, ON_DEMAND_PRICES)
    results = compare_commitments(WEIGHTS, RI_PRICES, on_demand_hourly)
    assert all(r["effective_hourly"] < on_demand_hourly for r in results)
    assert all(0 < r["breakeven_utilization_pct"] < 100 for r in results)


# --- load_keep_vms -------------------------------------------------------------


def test_load_keep_vms_filters_by_action(tmp_path: Path):
    path = tmp_path / "vm_recommendations.csv"
    _write_csv(
        path,
        ["instance", "actual_cpu", "pred_low", "pred_mid", "pred_high", "action", "risk", "monthly_savings"],
        [
            ["vm-1", "3.0", "2.9", "3.1", "3.5", "terminate", "safe", "0.5"],
            ["vm-2", "60.0", "58.0", "60.0", "62.0", "keep", "n/a", "0.0"],
            ["vm-3", "65.0", "63.0", "65.0", "67.0", "keep", "n/a", "0.0"],
        ],
    )
    keep = load_keep_vms(path)
    assert [v["instance"] for v in keep] == ["vm-2", "vm-3"]


def test_load_keep_vms_empty_when_none_kept(tmp_path: Path):
    path = tmp_path / "vm_recommendations.csv"
    _write_csv(
        path,
        ["instance", "actual_cpu", "pred_low", "pred_mid", "pred_high", "action", "risk", "monthly_savings"],
        [["vm-1", "3.0", "2.9", "3.1", "3.5", "terminate", "safe", "0.5"]],
    )
    assert load_keep_vms(path) == []


# --- build_ri_recommendations / write_ri_recommendations -----------------------


KEEP_VMS = [
    {"instance": "vm-2", "actual_cpu": "60.0"},
    {"instance": "vm-3", "actual_cpu": "65.0"},
]


def test_build_ri_recommendations_one_row_per_keep_vm():
    rows = build_ri_recommendations(KEEP_VMS, RI_PRICES, WEIGHTS)
    assert len(rows) == len(KEEP_VMS)
    assert {r["instance"] for r in rows} == {"vm-2", "vm-3"}


def test_build_ri_recommendations_picks_the_cheapest_commitment():
    rows = build_ri_recommendations(KEEP_VMS, RI_PRICES, WEIGHTS)
    assert all(r["recommended_commitment"] == "3yr All Upfront" for r in rows)


def test_build_ri_recommendations_savings_is_on_demand_minus_ri():
    rows = build_ri_recommendations(KEEP_VMS, RI_PRICES, WEIGHTS)
    for r in rows:
        assert r["monthly_savings"] == pytest.approx(
            r["on_demand_monthly_cost"] - r["ri_monthly_cost"], abs=0.01
        )


def test_build_ri_recommendations_never_invents_a_savings_plan_number():
    """Regression guard: no fabricated Savings Plans $ figure, ever."""
    rows = build_ri_recommendations(KEEP_VMS, RI_PRICES, WEIGHTS)
    for r in rows:
        assert "not computed" in r["savings_plan"]
        assert "$" not in r["savings_plan"]


def test_build_ri_recommendations_empty_keep_list_gives_empty_output():
    assert build_ri_recommendations([], RI_PRICES, WEIGHTS) == []


def test_build_ri_recommendations_raises_without_priceable_overlap():
    with pytest.raises(ValueError):
        build_ri_recommendations(KEEP_VMS, RI_PRICES, {"r5.8xlarge": 1.0})


def test_write_ri_recommendations_round_trips(tmp_path: Path):
    rows = build_ri_recommendations(KEEP_VMS, RI_PRICES, WEIGHTS)
    out_path = tmp_path / "ri_recommendations.csv"
    write_ri_recommendations(rows, out_path)

    with open(out_path) as f:
        written = list(csv.DictReader(f))

    assert len(written) == len(rows)
    assert written[0]["instance"] == rows[0]["instance"]
    assert set(written[0]) == {
        "instance", "actual_cpu", "on_demand_monthly_cost", "recommended_commitment",
        "ri_monthly_cost", "monthly_savings", "savings_pct", "breakeven_utilization_pct",
        "savings_plan",
    }


def test_hours_per_month_matches_project_convention():
    """730 hours/month is the constant used across pricing.py and predict.py."""
    assert HOURS_PER_MONTH == 730


def test_terms_are_whole_multiples_of_hours_per_month():
    assert TERMS["1yr"] == HOURS_PER_MONTH * 12
    assert TERMS["3yr"] == HOURS_PER_MONTH * 36

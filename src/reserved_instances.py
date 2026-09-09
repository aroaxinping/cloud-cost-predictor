"""Reserved Instance modeling for VMs classified "keep".

`src/predict.py` tags a VM "keep" when even the pessimistic (q=0.95) CPU
forecast clears the review threshold — these are the ~177 VMs this project
is NOT proposing to touch (no terminate/downsize/review action). That is
exactly the population worth reserving capacity for: only stable workloads
that are staying put are candidates for a 1-3 year commitment; anything
flagged terminate/downsize/review could disappear before the term is up.

This module compares their on-demand cost against 1yr/3yr EC2 Reserved
Instance commitments (No/Partial/All Upfront, standard offering class) using
real prices pulled from the AWS Bulk Pricing API (scripts/fetch_ri_pricing.py,
data/pricing/ec2_reserved_instances.json), and reports the breakeven
utilization: the minimum fraction of on-demand hours a VM must actually run
for the reservation to pay off.

Data limitation, read before trusting the numbers
---------------------------------------------------
The SAP dataset has no per-VM EC2 instance type - only aggregate
(ram_category, vcpu_category) counts in data/clean/vm_size_distribution.csv,
same as src/pricing.py uses for the fleet-level cost estimate. There is no
way to know which specific "keep" VM is a t3.micro vs an r5.8xlarge, so this
module prices all "keep" VMs the same way: a fleet-weighted average across
the EC2 types in the project's SIZE_TO_EC2 mix (see fleet_type_weights()),
which is the same approximation src/predict.py's load_fleet_avg_hourly()
already makes for the "n/a" savings shown for "keep" VMs in the README. Every
row in ri_recommendations.csv therefore carries the same on-demand/RI/savings
figures; only `instance` and `actual_cpu` are VM-specific. This is stated
explicitly rather than inventing a plausible-looking per-VM instance type.

Savings Plans
-------------
Not computed. AWS does not publish Savings Plans rates on the public,
unauthenticated Bulk Pricing API this project otherwise relies on for both
on-demand (scripts/fetch_ec2_pricing.py) and Reserved Instance
(scripts/fetch_ri_pricing.py) prices. Getting real Savings Plans rates needs
the authenticated `savingsplans:DescribeRates` API, which requires an AWS
account and credentials this pipeline does not have. Rather than approximate
with a guessed discount, every row's `savings_plan` column spells this out
(see SAVINGS_PLAN_NOTE below) so nobody mistakes a placeholder for a real
figure.
"""
import csv
import json
import logging
from pathlib import Path

from src.pricing import DATA, SIZE_TO_EC2, load_ec2_prices

logger = logging.getLogger(__name__)

RI_PRICING_FILE = Path(__file__).resolve().parent.parent / "data" / "pricing" / "ec2_reserved_instances.json"
RECOMMENDATIONS_FILE = DATA / "vm_recommendations.csv"
SIZE_DISTRIBUTION_FILE = DATA / "vm_size_distribution.csv"
OUTPUT_FILE = DATA / "ri_recommendations.csv"

HOURS_PER_MONTH = 730
TERMS: dict[str, int] = {"1yr": HOURS_PER_MONTH * 12, "3yr": HOURS_PER_MONTH * 12 * 3}
PAYMENT_OPTIONS = ("No Upfront", "Partial Upfront", "All Upfront")

SAVINGS_PLAN_NOTE = (
    "not computed: Savings Plans rates require the authenticated AWS "
    "savingsplans:DescribeRates API (AWS account + credentials), which is "
    "outside the public Bulk Pricing API this project uses for on-demand and "
    "Reserved Instance prices. No number is reported here rather than "
    "approximating one."
)


def load_ri_prices(path: Path | None = None) -> dict:
    """Load Reserved Instance rates from the pricing JSON (see scripts/fetch_ri_pricing.py)."""
    path = path or RI_PRICING_FILE
    with open(path) as f:
        data = json.load(f)
    logger.info("RI pricing: %s, verified %s", data["metadata"]["region"], data["metadata"]["verified"])
    return data["instances"]


def effective_hourly_rate(ri_entry: dict, hours_in_term: int) -> float:
    """Amortize the upfront fee across the term and add the recurring hourly charge.

    This is the standard way to compare a reservation to on-demand: spread
    whatever was paid upfront over every hour of the term, then add whatever
    is still billed hourly (0 for All Upfront, partial for Partial Upfront,
    the full on-demand-like rate for No Upfront).
    """
    return ri_entry["upfront_usd"] / hours_in_term + ri_entry["hourly_usd"]


def fleet_type_weights(path: Path | None = None) -> dict[str, float]:
    """Proportion of the fleet in each EC2 type, from vm_size_distribution.csv.

    Same source and method as src.pricing.estimate_fleet_cost(): the SAP
    dataset only has aggregate (ram_category, vcpu_category) counts, not a
    per-VM instance type, so this is the same fleet-level approximation used
    everywhere else in this project's cost estimates - reused here rather
    than reinvented.
    """
    path = path or SIZE_DISTRIBUTION_FILE
    with open(path) as f:
        sizes = list(csv.DictReader(f))
    total = sum(int(s["total_count"]) for s in sizes)
    if total == 0:
        raise ValueError(f"{path} has no rows")

    weights: dict[str, float] = {}
    for s in sizes:
        key = (s["ram_category"], s["vcpu_category"])
        ec2_type = SIZE_TO_EC2.get(key)
        if ec2_type is None:
            continue
        weights[ec2_type] = weights.get(ec2_type, 0.0) + int(s["total_count"]) / total
    return weights


def _weighted_average(rates_by_type: dict[str, float], weights: dict[str, float]) -> float:
    """Weighted mean over the types both dicts share, renormalized to those types."""
    covered = {t: w for t, w in weights.items() if t in rates_by_type}
    total_w = sum(covered.values())
    if total_w == 0:
        raise ValueError("No overlap between fleet type weights and priced instance types")
    return sum(rates_by_type[t] * w for t, w in covered.items()) / total_w


def fleet_weighted_on_demand_hourly(weights: dict[str, float], prices: dict | None = None) -> float:
    prices = prices if prices is not None else load_ec2_prices()
    hourly = {t: p["usd_per_hour"] for t, p in prices.items()}
    return _weighted_average(hourly, weights)


def fleet_weighted_ri_hourly(weights: dict[str, float], term: str, option: str, ri_prices: dict) -> float:
    hourly = {}
    for itype, by_term in ri_prices.items():
        entry = by_term.get(term, {}).get(option)
        if entry is None:
            continue
        hourly[itype] = effective_hourly_rate(entry, TERMS[term])
    return _weighted_average(hourly, weights)


def compare_commitments(weights: dict[str, float], ri_prices: dict, on_demand_hourly: float) -> list[dict]:
    """Effective hourly rate, savings and breakeven for every (term, payment option).

    breakeven_utilization_pct is the minimum fraction of on-demand hours a VM
    must actually run for the commitment to be cheaper than staying
    on-demand: ri_hourly / on_demand_hourly. Below that utilization,
    on-demand is still cheaper; above it, the commitment wins.
    """
    results = []
    for term in TERMS:
        for option in PAYMENT_OPTIONS:
            try:
                ri_hourly = fleet_weighted_ri_hourly(weights, term, option, ri_prices)
            except ValueError:
                continue
            results.append({
                "term": term,
                "payment_option": option,
                "effective_hourly": ri_hourly,
                "savings_pct": (on_demand_hourly - ri_hourly) / on_demand_hourly * 100,
                "breakeven_utilization_pct": ri_hourly / on_demand_hourly * 100,
            })
    results.sort(key=lambda r: r["effective_hourly"])
    return results


def load_keep_vms(path: Path | None = None) -> list[dict]:
    """VMs from vm_recommendations.csv with action == 'keep'."""
    path = path or RECOMMENDATIONS_FILE
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if r["action"] == "keep"]


def build_ri_recommendations(
    keep_vms: list[dict] | None = None,
    ri_prices: dict | None = None,
    weights: dict[str, float] | None = None,
    on_demand_hourly: float | None = None,
) -> list[dict]:
    """One row per 'keep' VM: on-demand vs the cheapest RI commitment.

    All dependencies are optional/injectable so tests can supply small fixed
    inputs instead of reading the real (123K-VM) data files.
    """
    if keep_vms is None:
        keep_vms = load_keep_vms()
    if ri_prices is None:
        ri_prices = load_ri_prices()
    if weights is None:
        weights = fleet_type_weights()
    if on_demand_hourly is None:
        on_demand_hourly = fleet_weighted_on_demand_hourly(weights)

    options = compare_commitments(weights, ri_prices, on_demand_hourly)
    if not options:
        raise ValueError("No Reserved Instance options priced for the fleet's instance-type mix")
    best = options[0]

    on_demand_monthly = round(on_demand_hourly * HOURS_PER_MONTH, 2)
    ri_monthly = round(best["effective_hourly"] * HOURS_PER_MONTH, 2)
    monthly_savings = round(on_demand_monthly - ri_monthly, 2)

    rows = []
    for vm in keep_vms:
        rows.append({
            "instance": vm["instance"],
            "actual_cpu": vm.get("actual_cpu", ""),
            "on_demand_monthly_cost": on_demand_monthly,
            "recommended_commitment": f"{best['term']} {best['payment_option']}",
            "ri_monthly_cost": ri_monthly,
            "monthly_savings": monthly_savings,
            "savings_pct": round(best["savings_pct"], 1),
            "breakeven_utilization_pct": round(best["breakeven_utilization_pct"], 1),
            "savings_plan": SAVINGS_PLAN_NOTE,
        })
    return rows


def write_ri_recommendations(rows: list[dict], out_path: Path | None = None) -> None:
    out_path = out_path or OUTPUT_FILE
    fields = [
        "instance", "actual_cpu", "on_demand_monthly_cost", "recommended_commitment",
        "ri_monthly_cost", "monthly_savings", "savings_pct", "breakeven_utilization_pct",
        "savings_plan",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="\n") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    logger.info("Wrote %s (%d rows)", out_path, len(rows))


def main() -> None:
    weights = fleet_type_weights()
    ri_prices = load_ri_prices()
    on_demand_hourly = fleet_weighted_on_demand_hourly(weights)
    options = compare_commitments(weights, ri_prices, on_demand_hourly)

    logger.info("Fleet-weighted on-demand rate: $%.4f/hr ($%.2f/mo per VM)",
                on_demand_hourly, on_demand_hourly * HOURS_PER_MONTH)
    logger.info("%-6s %-16s %12s %10s %10s", "Term", "Payment option", "$/mo", "Savings", "Breakeven")
    for o in options:
        logger.info("%-6s %-16s %12.2f %9.1f%% %9.1f%%",
                     o["term"], o["payment_option"], o["effective_hourly"] * HOURS_PER_MONTH,
                     o["savings_pct"], o["breakeven_utilization_pct"])

    keep_vms = load_keep_vms()
    logger.info("'keep' VMs (RI candidates): %d", len(keep_vms))

    rows = build_ri_recommendations(keep_vms, ri_prices, weights, on_demand_hourly)
    write_ri_recommendations(rows)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()

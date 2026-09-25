"""Map VM sizes to real AWS EC2 pricing and estimate waste in USD.

Prices are loaded from data/pricing/ec2_on_demand.json, which contains
verified on-demand rates from the AWS Bulk Pricing API.
"""
import csv
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DATA = Path(__file__).resolve().parent.parent / "data" / "clean"
PRICING_FILE = Path(__file__).resolve().parent.parent / "data" / "pricing" / "ec2_on_demand.json"

HOURS_PER_MONTH = 730

SIZE_TO_EC2: dict[tuple[str, str], str] = {
    ("Small", "Small"):              "t3.small",
    ("Medium", "Small"):             "t3.medium",
    ("Medium", "Medium"):            "m5.large",
    ("Medium", "Large"):             "m5.xlarge",
    ("Large", "Medium"):             "r5.large",
    ("Large", "Large"):              "r5.xlarge",
    ("Extra Large", "Medium"):       "r5.2xlarge",
    ("Extra Large", "Large"):        "r5.4xlarge",
    ("Extra Large", "Extra Large"):  "r5.8xlarge",
}

DOWNSIZE_TO: dict[str, str] = {
    "r5.8xlarge":  "r5.4xlarge",
    "r5.4xlarge":  "r5.2xlarge",
    "r5.2xlarge":  "r5.xlarge",
    "r5.xlarge":   "r5.large",
    "r5.large":    "m5.large",
    "m5.xlarge":   "m5.large",
    "m5.large":    "t3.medium",
    "t3.medium":   "t3.small",
    "t3.small":    "t3.micro",
}


def load_ec2_prices() -> dict:
    """Load EC2 on-demand prices from the pricing JSON."""
    with open(PRICING_FILE) as f:
        data = json.load(f)
    logger.info("EC2 pricing: %s, verified %s", data["metadata"]["region"], data["metadata"]["verified"])
    return data["instances"]


def get_price(prices: dict, instance_type: str) -> float:
    return prices[instance_type]["usd_per_hour"]


def _load_classification_rates() -> tuple[float, float]:
    """Read actual zombie/oversized rates from vm_classified.csv if available."""
    classified_path = DATA / "vm_classified.csv"
    if not classified_path.exists():
        logger.warning("vm_classified.csv not found, using default rates")
        return 0.109, 0.842
    with open(classified_path) as f:
        classes = [row["class"] for row in csv.DictReader(f)]
    n = len(classes)
    if n == 0:
        return 0.109, 0.842
    zombie_rate = sum(1 for c in classes if c == "zombie") / n
    downsize_rate = sum(1 for c in classes if c in ("idle", "oversized")) / n
    return zombie_rate, downsize_rate


def estimate_fleet_cost() -> None:
    """Load size distribution and compute monthly cost and savings potential."""
    prices = load_ec2_prices()
    zombie_rate, downsize_rate = _load_classification_rates()

    with open(DATA / "vm_size_distribution.csv") as f:
        sizes = list(csv.DictReader(f))

    total_snapshots = sum(int(s["total_count"]) for s in sizes)

    summary_path = DATA / "vm_utilization_summary.csv"
    if summary_path.exists():
        with open(summary_path) as f:
            fleet_size = sum(1 for _ in csv.DictReader(f))
    else:
        fleet_size = total_snapshots
        logger.warning("vm_utilization_summary.csv not found, using snapshot count as fleet size")

    results: list[dict] = []
    total_monthly = 0.0
    total_savings = 0.0

    for s in sizes:
        ram = s["ram_category"]
        vcpu = s["vcpu_category"]
        proportion = int(s["total_count"]) / total_snapshots
        vm_count = int(proportion * fleet_size)

        key = (ram, vcpu)
        if key not in SIZE_TO_EC2:
            continue

        ec2_type = SIZE_TO_EC2[key]
        hourly = get_price(prices, ec2_type)
        monthly_per_vm = hourly * HOURS_PER_MONTH
        monthly_total = monthly_per_vm * vm_count

        zombie_count = int(vm_count * zombie_rate)
        d_count = int(vm_count * downsize_rate)

        zombie_savings = zombie_count * monthly_per_vm

        downsize_savings = 0.0
        if ec2_type in DOWNSIZE_TO:
            smaller_type = DOWNSIZE_TO[ec2_type]
            smaller_hourly = get_price(prices, smaller_type)
            downsize_savings = d_count * (hourly - smaller_hourly) * HOURS_PER_MONTH

        total_monthly += monthly_total
        total_savings += zombie_savings + downsize_savings

        results.append({
            "size": f"{ram}/{vcpu}",
            "ec2_type": ec2_type,
            "vm_count": vm_count,
            "monthly_cost": round(monthly_total),
            "zombie_savings": round(zombie_savings),
            "downsize_savings": round(downsize_savings),
        })

    if total_monthly > 0:
        logger.info("Estimated monthly fleet cost: $%s", f"{total_monthly:,.0f}")
        logger.info("Potential monthly savings:    $%s (%d%%)",
                     f"{total_savings:,.0f}", total_savings / total_monthly * 100)
        logger.info("Per-VM average waste:         $%d/month", total_savings / fleet_size)

    for r in sorted(results, key=lambda x: -x["monthly_cost"]):
        logger.info("  %14s x %6s: $%10s/mo | save $%8s",
                     r["ec2_type"], f"{r['vm_count']:,}",
                     f"{r['monthly_cost']:,}",
                     f"{r['zombie_savings'] + r['downsize_savings']:,}")

    out_path = DATA / "fleet_cost_estimate.csv"
    with open(out_path, "w", newline="\n") as f:
        w = csv.DictWriter(f, fieldnames=["size", "ec2_type", "vm_count",
                                          "monthly_cost", "zombie_savings", "downsize_savings"])
        w.writeheader()
        w.writerows(results)
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    estimate_fleet_cost()


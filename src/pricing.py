"""Map VM sizes to real AWS EC2 pricing and estimate waste in USD.

Prices are loaded from data/pricing/ec2_on_demand.json, which contains
verified on-demand rates from the AWS Bulk Pricing API. Run
scripts/fetch_ec2_pricing.py to refresh them.
"""
import csv
import json
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data" / "clean"
PRICING_FILE = Path(__file__).resolve().parent.parent / "data" / "pricing" / "ec2_on_demand.json"

HOURS_PER_MONTH = 730


def load_ec2_prices():
    """Load EC2 on-demand prices from the pricing JSON."""
    with open(PRICING_FILE) as f:
        data = json.load(f)
    print(f"EC2 pricing: {data['metadata']['region']}, verified {data['metadata']['verified']}")
    return data["instances"]


def get_price(prices, instance_type):
    return prices[instance_type]["usd_per_hour"]


# SAP size categories -> EC2 instance type mapping
SIZE_TO_EC2 = {
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

# Right-sizing: if oversized, the next size down
DOWNSIZE_TO = {
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


def estimate_fleet_cost():
    """Load size distribution and compute monthly cost and savings potential."""
    prices = load_ec2_prices()

    with open(DATA / "vm_size_distribution.csv") as f:
        sizes = list(csv.DictReader(f))

    # The count column represents VM-snapshots (30 days x ~daily),
    # so we need to normalize. From ingest we know there are ~123K unique VMs.
    total_snapshots = sum(int(s["total_count"]) for s in sizes)

    results = []
    total_monthly = 0
    total_savings = 0

    for s in sizes:
        ram = s["ram_category"]
        vcpu = s["vcpu_category"]
        proportion = int(s["total_count"]) / total_snapshots
        vm_count = int(proportion * 123_363)

        key = (ram, vcpu)
        if key not in SIZE_TO_EC2:
            continue

        ec2_type = SIZE_TO_EC2[key]
        hourly = get_price(prices, ec2_type)
        info = prices[ec2_type]
        monthly_per_vm = hourly * HOURS_PER_MONTH
        monthly_total = monthly_per_vm * vm_count

        # Savings: zombies (10.9%) can be terminated, idle+oversized (84.2%) can be downsized
        zombie_count = int(vm_count * 0.109)
        downsize_count = int(vm_count * 0.842)

        zombie_savings = zombie_count * monthly_per_vm

        downsize_savings = 0
        if ec2_type in DOWNSIZE_TO:
            smaller_type = DOWNSIZE_TO[ec2_type]
            smaller_hourly = get_price(prices, smaller_type)
            downsize_savings = downsize_count * (hourly - smaller_hourly) * HOURS_PER_MONTH

        total_monthly += monthly_total
        total_savings += zombie_savings + downsize_savings

        results.append({
            "size": f"{ram}/{vcpu}",
            "ec2_type": ec2_type,
            "vm_count": vm_count,
            "vcpus": info["vcpus"],
            "ram_gb": info["ram_gb"],
            "hourly_usd": hourly,
            "monthly_cost": round(monthly_total),
            "zombie_savings": round(zombie_savings),
            "downsize_savings": round(downsize_savings),
        })

    print(f"\nEstimated monthly fleet cost: ${total_monthly:,.0f}")
    print(f"Potential monthly savings:    ${total_savings:,.0f} ({total_savings/total_monthly*100:.0f}%)")
    print(f"Per-VM average waste:         ${total_savings/123_363:.0f}/month")
    print()

    for r in sorted(results, key=lambda x: -x["monthly_cost"]):
        print(f"  {r['ec2_type']:>14} x {r['vm_count']:>6,}: "
              f"${r['monthly_cost']:>10,}/mo | "
              f"save ${r['zombie_savings'] + r['downsize_savings']:>8,}")

    out_path = DATA / "fleet_cost_estimate.csv"
    with open(out_path, "w", newline="\n") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    estimate_fleet_cost()

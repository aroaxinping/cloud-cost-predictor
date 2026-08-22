"""Fetch current EC2 on-demand pricing from the AWS Bulk Pricing API.

The AWS Price List API is public and requires no credentials.
This script queries it for the specific instance types used in the project
and updates data/pricing/ec2_on_demand.json.

Usage:
    python scripts/fetch_ec2_pricing.py [--region us-east-1]
"""
import argparse
import json
import urllib.request
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PRICING_FILE = ROOT / "data" / "pricing" / "ec2_on_demand.json"

INSTANCE_TYPES = [
    "t3.micro", "t3.small", "t3.medium",
    "m5.large", "m5.xlarge",
    "r5.large", "r5.xlarge", "r5.2xlarge", "r5.4xlarge", "r5.8xlarge",
]

# AWS Bulk Pricing API (public, no auth required)
OFFERS_INDEX = "https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonEC2/current/region_index.json"


def fetch_region_url(region):
    """Get the pricing file URL for a specific region."""
    print("Fetching region index...")
    with urllib.request.urlopen(OFFERS_INDEX, timeout=30) as resp:
        index = json.load(resp)
    region_data = index["regions"].get(region)
    if not region_data:
        raise ValueError(f"Region {region} not found. Available: {sorted(index['regions'])}")
    csv_url = "https://pricing.us-east-1.amazonaws.com" + region_data["currentVersionUrl"]
    return csv_url.replace(".json", ".csv")


def fetch_prices_from_csv(csv_url, instance_types):
    """Stream the EC2 pricing CSV and extract on-demand Linux prices.

    The CSV is large (~500MB) but we stream line-by-line and stop
    once all target instance types are found.
    """
    import csv
    import io

    print("Streaming pricing CSV (this may take a moment)...")
    prices = {}
    target = set(instance_types)

    req = urllib.request.Request(csv_url)
    with urllib.request.urlopen(req, timeout=120) as resp:
        reader = csv.reader(io.TextIOWrapper(resp, encoding="utf-8"))

        header = None
        for row in reader:
            if header is None:
                if "Instance Type" in row:
                    header = row
                    it_idx = header.index("Instance Type")
                    price_idx = header.index("PricePerUnit")
                    unit_idx = header.index("Unit")
                    os_idx = header.index("Operating System")
                    tenancy_idx = header.index("Tenancy")
                    preinstalled_idx = header.index("Pre Installed S/W")
                    cap_idx = header.index("CapacityStatus")
                    vcpu_idx = header.index("vCPU")
                    mem_idx = header.index("Memory")
                continue

            if len(row) <= max(it_idx, price_idx, os_idx):
                continue

            itype = row[it_idx]
            if itype not in target:
                continue

            # Filter: Linux, Shared tenancy, no pre-installed SW, Used capacity
            if (row[os_idx] != "Linux" or
                row[tenancy_idx] != "Shared" or
                row[preinstalled_idx] != "NA" or
                row[cap_idx] != "Used" or
                row[unit_idx] != "Hrs"):
                continue

            try:
                price = float(row[price_idx])
            except ValueError:
                continue

            if price <= 0:
                continue

            vcpus = int(row[vcpu_idx])
            ram_str = row[mem_idx].replace(" GiB", "").replace(",", "")
            ram_gb = float(ram_str)

            prices[itype] = {
                "vcpus": vcpus,
                "ram_gb": ram_gb,
                "usd_per_hour": price,
            }
            print(f"  {itype}: ${price}/hr ({vcpus} vCPU, {ram_gb} GB)")

            if len(prices) == len(target):
                break

    return prices


def save_pricing(prices, region):
    """Save prices to the project's pricing JSON."""
    output = {
        "metadata": {
            "source": "AWS EC2 On-Demand Pricing (Bulk API)",
            "region": region,
            "os": "Linux",
            "verified": str(date.today()),
            "url": "https://aws.amazon.com/ec2/pricing/on-demand/",
        },
        "instances": dict(sorted(prices.items())),
    }
    PRICING_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PRICING_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved {len(prices)} instance prices to {PRICING_FILE}")


def main():
    parser = argparse.ArgumentParser(description="Fetch EC2 on-demand pricing")
    parser.add_argument("--region", default="us-east-1", help="AWS region (default: us-east-1)")
    args = parser.parse_args()

    csv_url = fetch_region_url(args.region)
    prices = fetch_prices_from_csv(csv_url, INSTANCE_TYPES)

    missing = set(INSTANCE_TYPES) - set(prices)
    if missing:
        print(f"\nWarning: could not find prices for: {missing}")

    save_pricing(prices, args.region)


if __name__ == "__main__":
    main()

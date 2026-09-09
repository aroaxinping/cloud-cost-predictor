"""Fetch current EC2 Reserved Instance pricing from the AWS Bulk Pricing API.

Same public, unauthenticated endpoint as fetch_ec2_pricing.py (the AWS Price
List Bulk API), but this time we keep the "Reserved" TermType rows instead of
"OnDemand". Those rows carry LeaseContractLength (1yr/3yr), PurchaseOption
(No Upfront/Partial Upfront/All Upfront) and OfferingClass (standard/
convertible) — we keep "standard" RIs, which is what most fleets buy.

Savings Plans are NOT in this file. AWS does not publish Savings Plans rates
through the public Bulk Pricing API — they're only available through the
authenticated `savingsplans:DescribeRates` API (needs an AWS account and
credentials). See the "Savings Plans" note in README.md / reserved_instances.py
for what's missing and how to fill it in later.

Usage:
    python scripts/fetch_ri_pricing.py [--region us-east-1]
"""
import argparse
import json
import urllib.request
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PRICING_FILE = ROOT / "data" / "pricing" / "ec2_reserved_instances.json"

INSTANCE_TYPES = [
    "t3.micro", "t3.small", "t3.medium",
    "m5.large", "m5.xlarge",
    "r5.large", "r5.xlarge", "r5.2xlarge", "r5.4xlarge", "r5.8xlarge",
]

TERMS = ["1yr", "3yr"]
PAYMENT_OPTIONS = ["No Upfront", "Partial Upfront", "All Upfront"]
OFFERING_CLASS = "standard"

# Same AWS Bulk Pricing API used by fetch_ec2_pricing.py (public, no auth required)
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


def fetch_ri_prices_from_csv(csv_url, instance_types):
    """Stream the EC2 pricing CSV and extract standard Reserved Instance rates.

    For each (instance type, term, payment option) we need up to two rows:
    the recurring hourly charge (Unit == "Hrs") and, for Partial/All Upfront,
    the one-time upfront fee (Unit == "Quantity").
    """
    import csv
    import io

    print("Streaming pricing CSV for Reserved Instance rows (this may take a moment)...")
    raw: dict[tuple[str, str, str], dict[str, float]] = {}
    target = set(instance_types)

    req = urllib.request.Request(csv_url)
    with urllib.request.urlopen(req, timeout=300) as resp:
        reader = csv.reader(io.TextIOWrapper(resp, encoding="utf-8"))

        header = None
        for row in reader:
            if header is None:
                if "Instance Type" in row:
                    header = row
                    idx = {
                        name: header.index(name)
                        for name in (
                            "Instance Type", "TermType", "PricePerUnit", "Unit",
                            "Operating System", "Tenancy", "Pre Installed S/W",
                            "LeaseContractLength", "PurchaseOption", "OfferingClass",
                            "CapacityStatus",
                        )
                    }
                continue

            if len(row) <= max(idx.values()):
                continue

            itype = row[idx["Instance Type"]]
            if itype not in target:
                continue
            if row[idx["TermType"]] != "Reserved":
                continue
            if row[idx["OfferingClass"]] != OFFERING_CLASS:
                continue
            if (row[idx["Operating System"]] != "Linux"
                    or row[idx["Tenancy"]] != "Shared"
                    or row[idx["Pre Installed S/W"]] != "NA"):
                continue

            term = row[idx["LeaseContractLength"]]
            option = row[idx["PurchaseOption"]]
            if term not in TERMS or option not in PAYMENT_OPTIONS:
                continue

            try:
                price = float(row[idx["PricePerUnit"]])
            except ValueError:
                continue

            key = (itype, term, option)
            entry = raw.setdefault(key, {"hourly_usd": 0.0, "upfront_usd": 0.0})
            if row[idx["Unit"]] == "Hrs":
                entry["hourly_usd"] = price
            elif row[idx["Unit"]] == "Quantity":
                entry["upfront_usd"] = price

            if len(raw) == len(target) * len(TERMS) * len(PAYMENT_OPTIONS):
                # Every combo seen at least once (may still be missing the
                # second row for some); keep streaming a little further isn't
                # needed since rows for the same key are adjacent in this file.
                pass

    return raw


def reshape(raw: dict[tuple[str, str, str], dict[str, float]]) -> dict:
    """Turn the flat (type, term, option) dict into nested instance -> term -> option."""
    out: dict[str, dict] = {}
    for (itype, term, option), prices in raw.items():
        out.setdefault(itype, {}).setdefault(term, {})[option] = {
            "hourly_usd": round(prices["hourly_usd"], 4),
            "upfront_usd": round(prices["upfront_usd"], 2),
        }
    return out


def save_pricing(instances: dict, region: str) -> None:
    output = {
        "metadata": {
            "source": "AWS EC2 Reserved Instance Pricing, standard offering class (Bulk API)",
            "region": region,
            "os": "Linux",
            "verified": str(date.today()),
            "url": "https://aws.amazon.com/ec2/pricing/reserved-instances/pricing/",
            "note": (
                "Savings Plans rates are not published on the public Bulk Pricing "
                "API and are not included here. See README.md / "
                "src/reserved_instances.py for what would be needed to add them."
            ),
        },
        "instances": dict(sorted(instances.items())),
    }
    PRICING_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PRICING_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved Reserved Instance prices for {len(instances)} instance types to {PRICING_FILE}")


def main():
    parser = argparse.ArgumentParser(description="Fetch EC2 Reserved Instance pricing")
    parser.add_argument("--region", default="us-east-1", help="AWS region (default: us-east-1)")
    args = parser.parse_args()

    csv_url = fetch_region_url(args.region)
    raw = fetch_ri_prices_from_csv(csv_url, INSTANCE_TYPES)
    instances = reshape(raw)

    missing = set(INSTANCE_TYPES) - set(instances)
    if missing:
        print(f"\nWarning: could not find Reserved Instance prices for: {missing}")

    save_pricing(instances, args.region)


if __name__ == "__main__":
    main()

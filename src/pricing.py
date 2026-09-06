"""Map VM sizes to real AWS EC2 pricing and estimate waste in USD."""
import csv
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data" / "clean"

# Realistic EC2 pricing (us-east-1, on-demand, Linux, USD/hour) - Sep 2024
# Mapped from SAP size categories to plausible EC2 equivalents
SIZE_MAP = {
    ("Small", "Small"):       ("t3.small",    2, 2,   0.0208),
    ("Medium", "Small"):      ("t3.medium",   2, 4,   0.0416),
    ("Medium", "Medium"):     ("m5.large",    2, 8,   0.0960),
    ("Medium", "Large"):      ("m5.xlarge",   4, 16,  0.1920),
    ("Large", "Medium"):      ("r5.large",    2, 16,  0.1260),
    ("Large", "Large"):       ("r5.xlarge",   4, 32,  0.2520),
    ("Extra Large", "Medium"):("r5.2xlarge",  8, 64,  0.5040),
    ("Extra Large", "Large"): ("r5.4xlarge", 16, 128, 1.0080),
    ("Extra Large", "Extra Large"): ("r5.8xlarge", 32, 256, 2.0160),
}

# Right-sizing: if oversized, what could they move down to?
DOWNSIZE = {
    "r5.8xlarge":  ("r5.4xlarge", 1.0080),
    "r5.4xlarge":  ("r5.2xlarge", 0.5040),
    "r5.2xlarge":  ("r5.xlarge",  0.2520),
    "r5.xlarge":   ("r5.large",   0.1260),
    "r5.large":    ("m5.large",   0.0960),
    "m5.xlarge":   ("m5.large",   0.0960),
    "m5.large":    ("t3.medium",  0.0416),
    "t3.medium":   ("t3.small",   0.0208),
    "t3.small":    ("t3.micro",   0.0104),
}

HOURS_PER_MONTH = 730

def estimate_fleet_cost():
    """Load size distribution and compute monthly cost and savings potential."""
    with open(DATA / "vm_size_distribution.csv") as f:
        sizes = list(csv.DictReader(f))

    # The count column represents VM-snapshots (30 days × ~daily), 
    # so we need to normalize. From ingest we know there are ~123K unique VMs.
    # Let's compute proportions and apply to 123,363 VMs.
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
        if key not in SIZE_MAP:
            continue
        
        ec2_type, vcpus, ram_gb, hourly = SIZE_MAP[key]
        monthly_per_vm = hourly * HOURS_PER_MONTH
        monthly_total = monthly_per_vm * vm_count

        # Savings: zombies (10.9%) can be terminated, idle+oversized (84.2%) can be downsized
        zombie_count = int(vm_count * 0.109)
        downsize_count = int(vm_count * 0.842)
        
        zombie_savings = zombie_count * monthly_per_vm
        
        downsize_savings = 0
        if ec2_type in DOWNSIZE:
            _, smaller_hourly = DOWNSIZE[ec2_type]
            downsize_savings = downsize_count * (hourly - smaller_hourly) * HOURS_PER_MONTH

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

    print(f"Estimated monthly fleet cost: ${total_monthly:,.0f}")
    print(f"Potential monthly savings:    ${total_savings:,.0f} ({total_savings/total_monthly*100:.0f}%)")
    print(f"Per-VM average waste:         ${total_savings/123_363:.0f}/month")
    print()
    
    for r in sorted(results, key=lambda x: -x["monthly_cost"]):
        print(f"  {r['ec2_type']:>14} × {r['vm_count']:>6,}: "
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

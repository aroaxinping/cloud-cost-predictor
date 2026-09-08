"""EDA: fleet-level waste analysis and per-VM utilization profiles."""
import csv
from collections import Counter
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data" / "clean"


def classify_vm(cpu_mean: float, mem_mean: float, cpu_p95: float) -> str:
    """Classify a VM into an actionable bucket."""
    if cpu_mean < 5 and mem_mean < 20:
        return "zombie"
    if cpu_p95 < 10:
        return "idle"
    if cpu_mean < 20 and cpu_p95 < 50:
        return "oversized"
    if cpu_mean >= 20 and cpu_mean < 80:
        return "right-sized"
    if cpu_mean >= 80:
        return "hot"
    return "review"


def load_and_classify() -> list[dict]:
    """Load VM summary and add classification."""
    vms: list[dict] = []
    with open(DATA / "vm_utilization_summary.csv") as f:
        for row in csv.DictReader(f):
            vm: dict = {k: row[k] for k in ("instance",)}
            for k in ("cpu_mean", "cpu_median", "cpu_p5", "cpu_p95", "cpu_min", "cpu_max",
                       "mem_mean", "mem_median", "mem_p5", "mem_p95", "mem_min", "mem_max"):
                vm[k] = float(row[k])
            vm["cpu_n"] = int(row["cpu_n"])
            vm["mem_n"] = int(row["mem_n"])
            vm["class"] = classify_vm(vm["cpu_mean"], vm["mem_mean"], vm["cpu_p95"])
            vms.append(vm)
    return vms


def fleet_summary(vms: list[dict]) -> None:
    """Print fleet-level statistics."""
    n = len(vms)
    classes = Counter(v["class"] for v in vms)

    print(f"Fleet: {n:,} VMs")
    print()
    print("Classification:")
    for cls in ("zombie", "idle", "oversized", "right-sized", "hot", "review"):
        c = classes.get(cls, 0)
        print(f"  {cls:>12}: {c:>7,} ({c/n*100:5.1f}%)")

    waste_candidates = [v for v in vms if v["class"] in ("zombie", "idle", "oversized")]
    print(f"\nWaste candidates: {len(waste_candidates):,} ({len(waste_candidates)/n*100:.1f}%)")

    cpu_means = sorted(v["cpu_mean"] for v in vms)
    print(f"\nCPU utilization distribution:")
    for pct in (5, 25, 50, 75, 90, 95, 99):
        idx = min(len(cpu_means)-1, int(len(cpu_means)*pct/100))
        print(f"  P{pct:>2}: {cpu_means[idx]:6.1f}%")


def write_classified(vms: list[dict], out_path: Path | None = None) -> None:
    """Write classified VM data."""
    if out_path is None:
        out_path = DATA / "vm_classified.csv"
    fields = ["instance", "class",
              "cpu_mean", "cpu_median", "cpu_p5", "cpu_p95",
              "mem_mean", "mem_median", "mem_p5", "mem_p95"]
    with open(out_path, "w", newline="\n") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for vm in sorted(vms, key=lambda v: v["cpu_mean"]):
            row = {k: round(v, 2) if isinstance(v := vm[k], float) else v for k in fields}
            w.writerow(row)
    print(f"Wrote {out_path} ({len(vms)} rows)")


if __name__ == "__main__":
    vms = load_and_classify()
    fleet_summary(vms)
    write_classified(vms)

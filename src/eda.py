"""EDA: fleet-level waste analysis and per-VM utilization profiles."""
import csv
import logging
from collections import Counter
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "clean"
CONFIG_FILE = ROOT / "config.yaml"


def load_config() -> dict:
    with open(CONFIG_FILE) as f:
        return yaml.safe_load(f)


def classify_vm(cpu_mean: float, mem_mean: float, cpu_p95: float,
                thresholds: dict | None = None) -> str:
    """Classify a VM into an actionable bucket.

    Memory-aware: high memory usage prevents zombie/idle classification
    even when CPU is low, avoiding unsafe termination of memory-bound VMs.
    """
    if thresholds is None:
        thresholds = load_config()["classification"]

    t = thresholds
    if cpu_mean < t["zombie_cpu_max"] and mem_mean < t["zombie_mem_max"]:
        return "zombie"
    if cpu_mean >= t["hot_cpu_min"]:
        return "hot"
    if mem_mean >= t["memory_bound_min"]:
        return "right-sized"
    if cpu_p95 < t["idle_cpu_p95_max"] and mem_mean < t["idle_mem_max"]:
        return "idle"
    if (cpu_mean < t["oversized_cpu_mean_max"]
            and cpu_p95 < t["oversized_cpu_p95_max"]
            and mem_mean < t["oversized_mem_max"]):
        return "oversized"
    if cpu_mean >= t["oversized_cpu_mean_max"]:
        return "right-sized"
    return "review"


def load_and_classify(thresholds: dict | None = None) -> list[dict]:
    """Load VM summary and add classification."""
    if thresholds is None:
        thresholds = load_config()["classification"]

    vms: list[dict] = []
    with open(DATA / "vm_utilization_summary.csv") as f:
        for row in csv.DictReader(f):
            vm: dict = {k: row[k] for k in ("instance",)}
            for k in ("cpu_mean", "cpu_median", "cpu_p5", "cpu_p95", "cpu_min", "cpu_max",
                       "mem_mean", "mem_median", "mem_p5", "mem_p95", "mem_min", "mem_max"):
                vm[k] = float(row[k])
            vm["cpu_n"] = int(row["cpu_n"])
            vm["mem_n"] = int(row["mem_n"])
            vm["class"] = classify_vm(vm["cpu_mean"], vm["mem_mean"], vm["cpu_p95"], thresholds)
            vms.append(vm)
    return vms


def fleet_summary(vms: list[dict]) -> dict:
    """Compute and log fleet-level statistics. Returns summary dict."""
    n = len(vms)
    classes = Counter(v["class"] for v in vms)

    logger.info("Fleet: %s VMs", f"{n:,}")
    for cls in ("zombie", "idle", "oversized", "right-sized", "hot", "review"):
        c = classes.get(cls, 0)
        logger.info("  %12s: %7s (%5.1f%%)", cls, f"{c:,}", c / n * 100)

    waste_candidates = [v for v in vms if v["class"] in ("zombie", "idle", "oversized")]
    logger.info("Waste candidates: %s (%.1f%%)", f"{len(waste_candidates):,}",
                len(waste_candidates) / n * 100)

    return {"total": n, "classes": dict(classes), "waste_count": len(waste_candidates)}


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
    logger.info("Wrote %s (%d rows)", out_path, len(vms))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    vms = load_and_classify()
    fleet_summary(vms)
    write_classified(vms)

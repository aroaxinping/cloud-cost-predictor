"""Stream SAP dataset from zip and produce per-VM summary statistics."""
import csv
import io
import logging
import statistics
import zipfile
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

ZIP = Path(__file__).resolve().parent.parent / "data" / "raw" / "sap.zip"
PREFIX = "sap-cloud-infrastructure-dataset/data/"


def stream_metric(zf: zipfile.ZipFile, filename: str):
    """Yield (instance, value) from a vrops CSV inside the zip."""
    with zf.open(PREFIX + filename) as f:
        reader = csv.reader(io.TextIOWrapper(f, "utf-8"))
        header = next(reader)
        inst_idx = header.index("Instance")
        val_idx = header.index("Value")
        for row in reader:
            try:
                yield row[inst_idx], float(row[val_idx])
            except (ValueError, IndexError):
                continue


def summarize_metric(zf: zipfile.ZipFile, filename: str) -> dict:
    """Return {instance: {mean, median, p5, p95, min, max, n}}."""
    buckets: dict[str, list[float]] = defaultdict(list)
    count = 0
    for inst, val in stream_metric(zf, filename):
        buckets[inst].append(val)
        count += 1
        if count % 5_000_000 == 0:
            logger.info("  %s rows...", f"{count:,}")

    logger.info("  %s total rows, %s instances", f"{count:,}", f"{len(buckets):,}")
    result = {}
    for inst, vals in buckets.items():
        vals.sort()
        n = len(vals)
        result[inst] = {
            "mean": statistics.mean(vals),
            "std": statistics.stdev(vals) if n > 1 else 0.0,
            "median": statistics.median(vals),
            "p5": vals[max(0, int(n * 0.05))],
            "p95": vals[min(n - 1, int(n * 0.95))],
            "min": vals[0],
            "max": vals[-1],
            "n": n,
        }
    return result


def load_vm_sizes(zf: zipfile.ZipFile) -> dict[tuple[str, str], int]:
    """Return {(ram_cat, vcpu_cat): total_count} from vm_classification."""
    sizes: dict[tuple[str, str], int] = defaultdict(int)
    with zf.open(PREFIX + "vm_classification_30d.csv") as f:
        reader = csv.DictReader(io.TextIOWrapper(f, "utf-8"))
        for row in reader:
            key = (row["RAM_Category"], row["VCPU_Category"])
            sizes[key] += int(row["Count"])
    return sizes


def build(out_dir: Path | None = None) -> tuple[Path, Path]:
    if out_dir is None:
        out_dir = Path(__file__).resolve().parent.parent / "data" / "clean"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    zf = zipfile.ZipFile(ZIP)

    logger.info("Processing CPU usage...")
    cpu = summarize_metric(zf, "vrops_virtualmachine_cpu_usage_ratio_all.csv")

    logger.info("Processing memory usage...")
    mem = summarize_metric(zf, "vrops_virtualmachine_memory_usage_ratio_all.csv")

    all_instances = sorted(set(cpu) | set(mem))
    logger.info("Total unique VMs: %s", f"{len(all_instances):,}")

    out_path = out_dir / "vm_utilization_summary.csv"
    with open(out_path, "w", newline="\n") as f:
        w = csv.writer(f)
        w.writerow([
            "instance",
            "cpu_mean", "cpu_std", "cpu_median", "cpu_p5", "cpu_p95", "cpu_min", "cpu_max", "cpu_n",
            "mem_mean", "mem_std", "mem_median", "mem_p5", "mem_p95", "mem_min", "mem_max", "mem_n",
        ])
        for inst in all_instances:
            c = cpu.get(inst, {})
            m = mem.get(inst, {})
            w.writerow([
                inst,
                *[round(c.get(k, 0), 4) for k in ("mean", "std", "median", "p5", "p95", "min", "max")],
                c.get("n", 0),
                *[round(m.get(k, 0), 4) for k in ("mean", "std", "median", "p5", "p95", "min", "max")],
                m.get("n", 0),
            ])

    logger.info("Wrote %s (%d rows)", out_path, len(all_instances))

    logger.info("Processing VM sizes...")
    sizes = load_vm_sizes(zf)
    sizes_path = out_dir / "vm_size_distribution.csv"
    with open(sizes_path, "w", newline="\n") as f:
        w = csv.writer(f)
        w.writerow(["ram_category", "vcpu_category", "total_count"])
        for (ram, vcpu), count in sorted(sizes.items(), key=lambda x: -x[1]):
            w.writerow([ram, vcpu, count])
    logger.info("Wrote %s", sizes_path)

    zf.close()
    return out_path, sizes_path


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Ingest SAP dataset and produce VM summaries")
    parser.add_argument("--output-dir", default=None, help="output directory for clean CSVs")
    args = parser.parse_args()
    build(Path(args.output_dir) if args.output_dir else None)

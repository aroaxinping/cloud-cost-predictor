"""Stream SAP dataset from zip and produce per-VM summary statistics."""
import zipfile, io, csv, json, sys
from collections import defaultdict
from pathlib import Path
import statistics

ZIP = Path(__file__).resolve().parent.parent / "data" / "raw" / "sap.zip"
PREFIX = "sap-cloud-infrastructure-dataset/data/"

def stream_metric(zf, filename):
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

def summarize_metric(zf, filename):
    """Return {instance: {mean, median, p5, p95, min, max, n}}."""
    buckets = defaultdict(list)
    count = 0
    for inst, val in stream_metric(zf, filename):
        buckets[inst].append(val)
        count += 1
        if count % 5_000_000 == 0:
            print(f"  {count:,} rows...", file=sys.stderr)

    print(f"  {count:,} total rows, {len(buckets):,} instances", file=sys.stderr)
    result = {}
    for inst, vals in buckets.items():
        vals.sort()
        n = len(vals)
        result[inst] = {
            "mean": statistics.mean(vals),
            "median": statistics.median(vals),
            "p5": vals[max(0, int(n * 0.05))],
            "p95": vals[min(n - 1, int(n * 0.95))],
            "min": vals[0],
            "max": vals[-1],
            "n": n,
        }
    return result

def load_vm_sizes(zf):
    """Return {(ram_cat, vcpu_cat): total_count} from vm_classification."""
    sizes = defaultdict(int)
    with zf.open(PREFIX + "vm_classification_30d.csv") as f:
        reader = csv.DictReader(io.TextIOWrapper(f, "utf-8"))
        for row in reader:
            key = (row["RAM_Category"], row["VCPU_Category"])
            sizes[key] += int(row["Count"])
    return sizes

def build(out_dir=None):
    if out_dir is None:
        out_dir = Path(__file__).resolve().parent.parent / "data" / "clean"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    zf = zipfile.ZipFile(ZIP)

    print("Processing CPU usage...", file=sys.stderr)
    cpu = summarize_metric(zf, "vrops_virtualmachine_cpu_usage_ratio_all.csv")

    print("Processing memory usage...", file=sys.stderr)
    mem = summarize_metric(zf, "vrops_virtualmachine_memory_usage_ratio_all.csv")

    all_instances = sorted(set(cpu) | set(mem))
    print(f"Total unique VMs: {len(all_instances):,}", file=sys.stderr)

    out_path = out_dir / "vm_utilization_summary.csv"
    with open(out_path, "w", newline="\n") as f:
        w = csv.writer(f)
        w.writerow([
            "instance",
            "cpu_mean", "cpu_median", "cpu_p5", "cpu_p95", "cpu_min", "cpu_max", "cpu_n",
            "mem_mean", "mem_median", "mem_p5", "mem_p95", "mem_min", "mem_max", "mem_n",
        ])
        for inst in all_instances:
            c = cpu.get(inst, {})
            m = mem.get(inst, {})
            w.writerow([
                inst,
                *[round(c.get(k, 0), 4) for k in ("mean", "median", "p5", "p95", "min", "max")],
                c.get("n", 0),
                *[round(m.get(k, 0), 4) for k in ("mean", "median", "p5", "p95", "min", "max")],
                m.get("n", 0),
            ])

    print(f"Wrote {out_path} ({len(all_instances)} rows)", file=sys.stderr)

    print("Processing VM sizes...", file=sys.stderr)
    sizes = load_vm_sizes(zf)
    sizes_path = out_dir / "vm_size_distribution.csv"
    with open(sizes_path, "w", newline="\n") as f:
        w = csv.writer(f)
        w.writerow(["ram_category", "vcpu_category", "total_count"])
        for (ram, vcpu), count in sorted(sizes.items(), key=lambda x: -x[1]):
            w.writerow([ram, vcpu, count])
    print(f"Wrote {sizes_path}", file=sys.stderr)

    zf.close()
    return out_path, sizes_path

if __name__ == "__main__":
    build()

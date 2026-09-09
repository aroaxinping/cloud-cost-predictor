"""Validate clean data files before they enter the pipeline."""
import csv
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "clean"

SUMMARY_REQUIRED_COLS = {
    "instance", "cpu_mean", "cpu_median",
    "cpu_p5", "cpu_p95", "cpu_min", "cpu_max", "cpu_n",
    "mem_mean", "mem_median",
    "mem_p5", "mem_p95", "mem_min", "mem_max", "mem_n",
}

NUMERIC_COLS = {
    "cpu_mean", "cpu_median", "cpu_p5", "cpu_p95",
    "cpu_min", "cpu_max", "mem_mean", "mem_median",
    "mem_p5", "mem_p95", "mem_min", "mem_max",
}


def validate_summary(path: Path | None = None) -> list[str]:
    """Check vm_utilization_summary.csv for schema and value issues."""
    if path is None:
        path = DATA_DIR / "vm_utilization_summary.csv"

    errors: list[str] = []

    if not path.exists():
        return [f"File not found: {path}"]

    with open(path) as f:
        reader = csv.DictReader(f)
        headers = set(reader.fieldnames or [])

        missing = SUMMARY_REQUIRED_COLS - headers
        if missing:
            errors.append(f"Missing columns: {sorted(missing)}")
            return errors

        seen_instances: set[str] = set()
        row_count = 0

        for i, row in enumerate(reader, start=2):
            row_count += 1
            inst = row["instance"]

            if not inst.strip():
                errors.append(f"Row {i}: empty instance ID")
                continue

            if inst in seen_instances:
                errors.append(f"Row {i}: duplicate instance '{inst}'")
            seen_instances.add(inst)

            for col in NUMERIC_COLS:
                try:
                    val = float(row[col])
                except (ValueError, TypeError):
                    errors.append(f"Row {i} ({inst}): non-numeric {col}={row[col]!r}")
                    continue

                if val < 0:
                    errors.append(f"Row {i} ({inst}): negative {col}={val}")

                if col not in ("cpu_std", "cpu_n", "mem_std", "mem_n") and val > 100:
                    errors.append(f"Row {i} ({inst}): {col}={val} exceeds 100%")

    if row_count == 0:
        errors.append("File is empty (no data rows)")

    return errors


def main():
    errors = validate_summary()
    if errors:
        print(f"VALIDATION FAILED — {len(errors)} issue(s):", file=sys.stderr)
        for e in errors[:20]:
            print(f"  - {e}", file=sys.stderr)
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more", file=sys.stderr)
        sys.exit(1)
    else:
        print("Validation passed.")


if __name__ == "__main__":
    main()

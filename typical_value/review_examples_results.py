"""Task 1.1 — Review the example-design results CSV for correctness.

Prints a structured report of data-integrity checks. Exits non-zero if any
check fails.

Usage:
    python review_examples_results.py [path_to_results_csv]
"""

from __future__ import annotations

import sys
from pathlib import Path

from data_checks import check_all, load_results

DEFAULT_CSV = Path(__file__).resolve().parent / "examples_eq_results.csv"


def main() -> int:
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CSV
    df = load_results(csv_path)

    print(f"Reviewing: {csv_path}")
    print(f"  rows: {len(df)}, columns: {len(df.columns)}")
    print()

    problems = check_all(df)
    if not problems:
        print("ALL CHECKS PASSED")
        return 0

    print(f"FAILED — {len(problems)} problem(s):")
    for p in problems:
        print(f"  - {p}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

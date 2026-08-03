"""
Example-design equivalent circuit calculator based on EquivalentCircuitParameters Rev.5.xlsx.

Usage:
    python examples_calculator.py examples_input.csv examples_eq_results.csv

Optional Excel output:
    python examples_calculator.py examples_input.csv examples_eq_results.csv --xlsx examples_eq_results.xlsx

Notes:
- Jconn = 3 for Delta; otherwise 1.
- Xcorr = 2/3 for PoleSpeed <= 4; otherwise 1/2.
- Inputs are expected to already be numeric, including BDTCorelFactor.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Mapping, Any

import pandas as pd

REQUIRED_COLUMNS = [
    "PoleSpeed",
    "Voltage",
    "Frequency",
    "Amps",
    "NoLoadAmps",
    "Connection",
    "WindingResistAt105",
    "BarRes",
    "ERRes",
    "BDTCorelFactor",
    "ReactCoilEndRun",
    "ResTotalStart",
    "ReactTotalRun",
    "ReactStatorSlotRun",
    "ReactStatorSlotStart",
    "ReactRotorSlotRun",
    "ReactRotorSlotStart",
    "ReactZigZagRun",
    "ReactZigZagStart",
    "ReactBeltLeakRun",
    "ReactTotalStart",
]

OUTPUT_COLUMNS = [
    "Zbase",
    "R1",
    "R2",
    "R3",
    "X1",
    "X2",
    "X3",
    "Xm",
    "XdPrime",
    "XdDoublePrime",
    "ShortCircuitTimeConstant",
    "SubTransientTimeConstant",
    "OpenCircuitTimeConstant",
    "L1",
    "L2",
    "L3",
    "FirstCycleInrush",
]


def _f(row: Mapping[str, Any], name: str) -> float:
    return float(row[name])


def parallel(a: float, b: float) -> float:
    return (a * b) / (a + b)


def parallel3(a: float, b: float, c: float) -> float:
    return (a * b * c) / ((a * b) + (b * c) + (c * a))


def calculate_row(row: Mapping[str, Any]) -> Dict[str, float]:
    """Calculate equivalent circuit parameters for one design row."""
    connection = str(row["Connection"]).strip().lower()
    jconn = 3.0 if connection.startswith("delta") else 1.0
    xcorr = (2.0 / 3.0) if _f(row, "PoleSpeed") <= 4.0 else 0.5
    omega = 2.0 * math.pi * _f(row, "Frequency")

    r1 = _f(row, "WindingResistAt105") / jconn
    r2 = (_f(row, "BarRes") + _f(row, "ERRes")) / jconn
    r3 = _f(row, "ResTotalStart") / jconn - r1

    x_total_run_corrected = _f(row, "ReactTotalRun") / _f(row, "BDTCorelFactor")

    x1_raw = (
        _f(row, "ReactStatorSlotRun")
        + 0.5 * _f(row, "ReactZigZagRun")
        + 0.5 * _f(row, "ReactBeltLeakRun")
        + xcorr * _f(row, "ReactCoilEndRun")
    )
    x2_raw = (
        _f(row, "ReactRotorSlotRun")
        + 0.5 * _f(row, "ReactZigZagRun")
        + 0.5 * _f(row, "ReactBeltLeakRun")
        + (1.0 - xcorr) * _f(row, "ReactCoilEndRun")
    )

    x1 = x_total_run_corrected * (x1_raw / (x1_raw + x2_raw)) / jconn
    x2 = x_total_run_corrected * (x2_raw / (x1_raw + x2_raw)) / jconn

    x1_start_raw = (
        _f(row, "ReactStatorSlotStart")
        + 0.5 * _f(row, "ReactZigZagStart")
        + xcorr * _f(row, "ReactCoilEndRun")
    ) / jconn
    x2_start_raw = (
        _f(row, "ReactRotorSlotStart")
        + 0.5 * _f(row, "ReactZigZagStart")
        + (1.0 - xcorr) * _f(row, "ReactCoilEndRun")
    ) / jconn

    x3 = _f(row, "ReactTotalStart") * (x2_start_raw / (x1_start_raw + x2_start_raw)) / jconn
    xm = _f(row, "Voltage") / (math.sqrt(3.0) * _f(row, "NoLoadAmps"))
    zbase = _f(row, "Voltage") / (math.sqrt(3.0) * _f(row, "Amps"))

    xd_prime = x1 + parallel(x2, xm)
    xd_double_prime = x1 + parallel3(x2, xm, x3)

    short_circuit_tc = (x2 + parallel(x1, xm)) / (omega * r2)
    sub_transient_tc = (x3 + parallel3(x1, x2, xm)) / (omega * r3)
    open_circuit_tc = (xm + x2) / (omega * r2)

    l1 = x1 / omega
    l2 = x2 / omega
    l3 = x3 / omega

    first_cycle_inrush = 1.25 * math.sqrt(2.0) * (1.0 + math.exp((-math.pi * r3) / x3))

    return {
        "Zbase": zbase,
        "R1": r1,
        "R2": r2,
        "R3": r3,
        "X1": x1,
        "X2": x2,
        "X3": x3,
        "Xm": xm,
        "XdPrime": xd_prime,
        "XdDoublePrime": xd_double_prime,
        "ShortCircuitTimeConstant": short_circuit_tc,
        "SubTransientTimeConstant": sub_transient_tc,
        "OpenCircuitTimeConstant": open_circuit_tc,
        "L1": l1,
        "L2": l2,
        "L3": l3,
        "FirstCycleInrush": first_cycle_inrush,
    }


def calculate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Append equivalent circuit outputs to a dataframe."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")

    results = pd.DataFrame([calculate_row(row) for row in df.to_dict("records")], index=df.index)
    return pd.concat([df.copy(), results[OUTPUT_COLUMNS]], axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate equivalent circuit parameters for each CSV row.")
    parser.add_argument("input_csv", help="Input CSV path")
    parser.add_argument("output_csv", help="Output CSV path")
    parser.add_argument("--xlsx", help="Optional Excel output path", default=None)
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    output_csv = Path(args.output_csv)

    df = pd.read_csv(input_path)
    out = calculate_dataframe(df)
    out.to_csv(output_csv, index=False)

    if args.xlsx:
        out.to_excel(Path(args.xlsx), index=False)

    print(f"Processed {len(out)} rows")
    print(f"Wrote {output_csv}")
    if args.xlsx:
        print(f"Wrote {args.xlsx}")


if __name__ == "__main__":
    main()

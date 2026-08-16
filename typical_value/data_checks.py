"""Reusable data-integrity checks for the example-design equivalent circuit dataset.

Used by:
- ``review_examples_results.py``  (Task 1.1 — review report)
- ``tests/test_data_integrity.py`` (Task 1.2 — regression/integrity tests)

Dataset: ``data/eq_parameters.csv`` — the 4,195-row design export with the
equivalent-circuit output columns appended by ``eq_calculator.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_NUM_ROWS = 4195

INPUT_COLUMNS = [
    "RatingID",
    "PoleSpeed",
    "MotorRotorInertia",
    "Connection",
    "BarMaterial",
    "ERMaterial",
    "WindingResistAt105",
    "BarRes",
    "ERRes",
    "StatorOD",
    "RotorOD",
    "RotorSlotType",
    "StatorSlotType",
    "NemaDesign",
    "NemaKVACode",
    "Name",
    "SRPM",
    "Efficiency100",
    "PowerFactor100",
    "HorsePower",
    "Voltage",
    "Frequency",
    "RPM",
    "Amps",
    "NoLoadAmps",
    "LockedRotorAmps",
    "LockedRotorTorque",
    "BreakDownTorque",
    "SFRiseByRes1",
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
    "FrameSize",
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

POSITIVE_COLUMNS = [
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
]

# Plausible band for the LR inrush factor: 1.25*sqrt(2)*(1+exp(-pi*r3/x3)) lies in (1.767, 3.536).
INRUSH_MIN, INRUSH_MAX = 1.75, 3.55


def load_results(path: Path | str) -> pd.DataFrame:
    return pd.read_csv(path)


def check_columns(df: pd.DataFrame) -> list[str]:
    expected = INPUT_COLUMNS + OUTPUT_COLUMNS
    missing = [c for c in expected if c not in df.columns]
    extra = [c for c in df.columns if c not in expected]
    problems = []
    if missing:
        problems.append(f"missing columns: {missing}")
    if extra:
        problems.append(f"unexpected columns: {extra}")
    return problems


def check_row_count(df: pd.DataFrame) -> list[str]:
    problems = []
    if len(df) != EXPECTED_NUM_ROWS:
        problems.append(f"row count {len(df)} != expected {EXPECTED_NUM_ROWS}")
    return problems


def check_ids(df: pd.DataFrame) -> list[str]:
    # RatingID is a rating (product) code, not a design-unique id: the dataset
    # legitimately holds several design rows per RatingID, so only nulls are
    # flagged here (duplicate rows are covered by the regeneration test).
    problems = []
    if df["RatingID"].isna().any():
        problems.append("null RatingID present")
    return problems


def check_no_nan_inf(df: pd.DataFrame) -> list[str]:
    problems = []
    num = df.select_dtypes(include=[np.number])
    if int(num.isna().sum().sum()) > 0:
        null_cols = num.columns[num.isna().any()].tolist()
        problems.append(f"null values in: {null_cols}")
    if int(np.isinf(num.to_numpy(dtype=float)).sum()) > 0:
        inf_cols = num.columns[np.isinf(num.to_numpy(dtype=float)).any(axis=0)].tolist()
        problems.append(f"infinite values in: {inf_cols}")
    return problems


def check_positive(df: pd.DataFrame) -> list[str]:
    problems = []
    for col in POSITIVE_COLUMNS:
        bad = (df[col] <= 0)
        n = int(bad.sum())
        if n:
            ids = df.loc[bad, "RatingID"].head(10).tolist()
            problems.append(f"{col} <= 0 in {n} rows (e.g. {ids})")
    return problems


def check_alpha_split(df: pd.DataFrame) -> list[str]:
    """X1/(X1+X2) must lie strictly within (0, 1)."""
    alpha = df["X1"] / (df["X1"] + df["X2"])
    bad = (alpha <= 0) | (alpha >= 1)
    problems = []
    if bad.any():
        ids = df.loc[bad, "RatingID"].tolist()
        problems.append(f"alpha = X1/(X1+X2) out of (0,1) in {int(bad.sum())} rows: {ids}")
    return problems


def check_xm_dominance(df: pd.DataFrame) -> list[str]:
    """Magnetising reactance must dominate total leakage: Xm > X1 + X2."""
    leak = df["X1"] + df["X2"]
    bad = df["Xm"] <= leak
    problems = []
    if bad.any():
        ids = df.loc[bad, "RatingID"].tolist()
        problems.append(f"Xm <= X1+X2 in {int(bad.sum())} rows: {ids}")
    return problems


def check_inrush_band(df: pd.DataFrame) -> list[str]:
    bad = (df["FirstCycleInrush"] < INRUSH_MIN) | (df["FirstCycleInrush"] > INRUSH_MAX)
    problems = []
    if bad.any():
        ids = df.loc[bad, "RatingID"].tolist()
        problems.append(
            f"FirstCycleInrush outside [{INRUSH_MIN}, {INRUSH_MAX}] in {int(bad.sum())} rows: {ids}"
        )
    return problems


def check_identities(df: pd.DataFrame, rtol: float = 1e-7, atol: float = 1e-6) -> list[str]:
    """Structural identities the calculator must satisfy.

    The committed CSV stores values to ~10 significant digits, so a small
    relative tolerance absorbs the rounding while still catching real
    structural violations.
    """
    problems = []
    conn = df["Connection"].str.lower().str.strip()
    jconn = np.where(conn.str.startswith("delta"), 3.0, 1.0)

    r1 = df["WindingResistAt105"] / jconn
    if not np.allclose(df["R1"], r1, rtol=rtol, atol=atol):
        problems.append("R1 != WindingResistAt105 / Jconn")

    zbase = df["Voltage"] / (np.sqrt(3.0) * df["Amps"])
    if not np.allclose(df["Zbase"], zbase, rtol=rtol, atol=atol):
        problems.append("Zbase != V_LL / (sqrt(3) * I_FL)")

    xm = df["Voltage"] / (np.sqrt(3.0) * df["NoLoadAmps"])
    if not np.allclose(df["Xm"], xm, rtol=rtol, atol=atol):
        problems.append("Xm != V_LL / (sqrt(3) * I_no-load)")

    return problems


def check_all(df: pd.DataFrame) -> list[str]:
    problems: list[str] = []
    for check in (
        check_columns,
        check_row_count,
        check_ids,
        check_no_nan_inf,
        check_positive,
        check_alpha_split,
        check_xm_dominance,
        check_inrush_band,
        check_identities,
    ):
        problems.extend(f"{check.__name__}: {p}" for p in check(df))
    return problems

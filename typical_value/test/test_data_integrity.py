"""Task 1.2 — Data-integrity tests for the example-design dataset.

These guard `examples_calculator.py` and the CSVs so future changes cannot
silently corrupt the data.

Run from the repo root (or with pytest from `typical_value/`):
    python -m pytest typical_value/test/test_data_integrity.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_checks import (  # noqa: E402
    EXPECTED_NUM_ROWS,
    check_alpha_split,
    check_columns,
    check_ids,
    check_identities,
    check_inrush_band,
    check_no_nan_inf,
    check_positive,
    check_row_count,
    check_xm_dominance,
)
from typical_value.eq_calculator import calculate_dataframe  # noqa: E402

RESULTS_CSV = Path(__file__).resolve().parent.parent / "examples_eq_results.csv"
INPUT_CSV = Path(__file__).resolve().parent.parent / "examples_input.csv"


def test_columns_match_expected_headers():
    df = pd.read_csv(RESULTS_CSV)
    assert check_columns(df) == []


def test_row_count():
    df = pd.read_csv(RESULTS_CSV)
    assert check_row_count(df) == []
    assert len(df) == EXPECTED_NUM_ROWS


def test_design_audit_ids_unique():
    df = pd.read_csv(RESULTS_CSV)
    assert check_ids(df) == []


def test_no_nan_or_infinite():
    df = pd.read_csv(RESULTS_CSV)
    assert check_no_nan_inf(df) == []


def test_all_positive():
    df = pd.read_csv(RESULTS_CSV)
    assert check_positive(df) == []


def test_alpha_split_in_unit_interval():
    df = pd.read_csv(RESULTS_CSV)
    assert check_alpha_split(df) == []


def test_xm_dominates_leakage():
    df = pd.read_csv(RESULTS_CSV)
    assert check_xm_dominance(df) == []


def test_inrush_in_physical_band():
    df = pd.read_csv(RESULTS_CSV)
    assert check_inrush_band(df) == []


def test_structural_identities():
    df = pd.read_csv(RESULTS_CSV)
    assert check_identities(df) == []


def test_deterministic_regeneration():
    """Re-running the calculator must reproduce the committed results file."""
    df_in = pd.read_csv(INPUT_CSV)
    recomputed = calculate_dataframe(df_in)
    committed = pd.read_csv(RESULTS_CSV)

    assert list(recomputed.columns) == list(committed.columns)

    num_cols = [c for c in committed.columns if c not in ("RatingID", "Connection", "FrameSize")]
    a = recomputed[num_cols].astype(float).to_numpy()
    b = committed[num_cols].astype(float).to_numpy()
    assert a.shape == b.shape
    assert np.allclose(a, b, rtol=1e-9, atol=1e-9), "regeneration differs from committed results"

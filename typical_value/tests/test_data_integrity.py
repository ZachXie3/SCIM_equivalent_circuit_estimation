"""Task 1.2 — Data-integrity tests for the example-design dataset.

These guard `eq_calculator.py` and the CSVs so future changes cannot
silently corrupt the data.

Run from the repo root (or with pytest from `typical_value/`):
    python -m pytest typical_value/tests/test_data_integrity.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_TV = Path(__file__).resolve().parent.parent
_ROOT = _TV.parent
sys.path.insert(0, str(_TV))
sys.path.insert(0, str(_ROOT))

from data_checks import (  # noqa: E402
    EXPECTED_NUM_ROWS,
    OUTPUT_COLUMNS,
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

RESULTS_CSV = _TV / "data" / "eq_parameters.csv"
INPUT_CSV = _TV / "data" / "eq_raw.csv"

# The committed eq_parameters.csv rounds L1/L2/L3 to ~3 significant digits
# (an older export), so the regeneration comparison uses a looser tolerance
# for those diagnostic columns only.
L_ROUNDED_TOL = 1e-2
L_COLS = ("L1", "L2", "L3")


def test_columns_match_expected_headers():
    df = pd.read_csv(RESULTS_CSV)
    assert check_columns(df) == []


def test_row_count():
    df = pd.read_csv(RESULTS_CSV)
    assert check_row_count(df) == []
    assert len(df) == EXPECTED_NUM_ROWS


def test_ids_present_no_nulls():
    # RatingID is a rating code shared by several design rows; only nulls are
    # checked (uniqueness is not guaranteed by the schema).
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

    # Committed file was exported before LockedRotorAmps2 was added to the raw
    # export; it must be a strict subset of the regenerated columns.
    assert set(committed.columns).issubset(set(recomputed.columns))
    for c in OUTPUT_COLUMNS:
        rtol = L_ROUNDED_TOL if c in L_COLS else 1e-9
        assert np.allclose(recomputed[c].to_numpy(), committed[c].to_numpy(),
                           rtol=rtol, atol=1e-9), f"{c} differs from committed"

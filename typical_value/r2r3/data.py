"""Dataset loading and derived columns for the R2/R3 analysis.

The source file is ``typical_value/data/eq_parameters.csv`` — the 4,195-row
design export (see ``SQLQuery.sql``) with the equivalent-circuit output
columns appended by ``eq_calculator.py``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_CSV = Path(__file__).resolve().parent.parent / "data" / "eq_parameters.csv"


def load_results(path: Path | str = DEFAULT_CSV) -> pd.DataFrame:
    return pd.read_csv(path)


def add_slip(df: pd.DataFrame) -> pd.DataFrame:
    """Add the full-load slip column ``s_fl`` from nameplate values."""
    ns = 120.0 * df["Frequency"] / df["PoleSpeed"]
    return df.assign(s_fl=(ns - df["RPM"]) / ns)


def valid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Rows usable for fitting: s_fl > 0 and HP > 0."""
    return df[(df["s_fl"] > 0) & (df["HorsePower"] > 0)]

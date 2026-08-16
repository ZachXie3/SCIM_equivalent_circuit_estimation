"""Constant ``R2/R3`` per bin — the approved slip-segment table method
(plan.md §2.7 / §11.1) plus the grouping-constant exploration (§6).

This is one of the R2/R3 analysis methods. Like ``models.py`` it exposes a
uniform fit / predict / score interface:

    fit_slip_segments(df)           -> params      (one constant ratio per bin)
    predict_slip_segments(params, s_fl) -> pd.Series   (per-row ratio)
    predict_slip_segments_r2(params, df) -> pd.Series  (predicted R2)
    score_slip_segments(df)         -> metrics dict (overall + per-bin bands)

Design (plan.md §2.7, "Approved alternative"):

    * one constant ``R2/R3`` per full-load-slip band — no exponent math, so a
      practising engineer can apply it by hand;
    * ``s_fl >= 5%`` is taken as ``1.0`` (trend extrapolation; only 12 rows
      above 5 % in the dataset);
    * overall accuracy is roughly that of the slip-only power law (Model B)
      and ~7 pp behind the continuous slip+HP formula.

The grouping-constant exploration (§6) is kept for reference: it shows that
a constant geometric-mean ratio per (voltage / pole / HP bucket / slip bin)
group loses to any slip-based model unless the grouping is very fine.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

# Approved slip-band edges (plan.md §2.7 table). Bands are half-open [lo, hi).
SLIP_BAND_EDGES = [0, 0.005, 0.01, 0.02, 0.03, 0.05, 1.0]

# s_fl >= EXTRAPOLATE_AT is assigned ratio 1.0 (upward-trend extrapolation).
EXTRAPOLATE_AT = 0.05


# ---------------------------------------------------------------------------
# Slip-segment table method (fit / predict / score)
# ---------------------------------------------------------------------------


def fit_slip_segments(
    df: pd.DataFrame,
    edges: list[float] | None = None,
) -> dict:
    """One geometric-mean ``R2/R3`` per slip band.

    Returns ``{"edges": [...], "ratios": {Interval: ratio}}``. Bands at or
    above ``EXTRAPOLATE_AT`` get ratio ``1.0``.
    """
    edges = list(edges if edges is not None else SLIP_BAND_EDGES)
    ratio = df["R2"] / df["R3"]
    cats = pd.cut(df["s_fl"], edges, right=False)
    gm = df.groupby(cats, observed=True).apply(
        lambda g: np.exp(np.log(g["R2"] / g["R3"]).mean())
    ).to_dict()
    ratios = {k: (1.0 if k.left >= EXTRAPOLATE_AT else v) for k, v in gm.items()}
    return {"edges": edges, "ratios": ratios}


def predict_slip_segments(params: Mapping, s_fl) -> pd.Series:
    """Map a slip array/series to the per-bin ``R2/R3`` ratio (as float)."""
    edges = np.asarray(params["edges"], dtype=float)
    ratios = dict(params["ratios"])
    ordered = sorted(ratios, key=lambda iv: iv.left)
    ratio_vals = np.array([ratios[k] for k in ordered], dtype=float)
    s = np.asarray(s_fl, dtype=float)
    # np.digitize(edges, right=False): band i is edges[i-1] <= s < edges[i],
    # so index = digitize - 1; values above the top edge clamp to the last
    # (extrapolated) band.
    idx = np.clip(np.digitize(s, edges, right=False) - 1, 0, len(ratio_vals) - 1)
    index = s_fl.index if isinstance(s_fl, pd.Series) else None
    return pd.Series(ratio_vals[idx], index=index)


def predict_slip_segments_r2(params: Mapping, df: pd.DataFrame) -> pd.Series:
    """Predicted ``R2 = ratio(s_fl) * R3``."""
    return predict_slip_segments(params, df["s_fl"]) * df["R3"]


def _band_rows(df: pd.DataFrame, params: Mapping) -> list[dict]:
    """Per-band table rows: label, n, ratio, bin RMSE, bin P10/P90 (%)."""
    pred = predict_slip_segments(params, df["s_fl"]).to_numpy()
    true = (df["R2"] / df["R3"]).to_numpy()
    rel_t = (pred - true) / true * 100.0
    cats = pd.cut(df["s_fl"], params["edges"], right=False)

    rows = []
    for k, r in dict(params["ratios"]).items():
        mask = (cats == k).to_numpy()
        label = f"{k.left*100:g}-{k.right*100:g}%"
        if k.left == 0.0:
            label = f"< {k.right*100:g}%"
        if k.right >= 1.0:
            label = f">= {k.left*100:g}%"
        seg = rel_t[mask]
        rows.append({
            "label": label,
            "n": int(mask.sum()),
            "ratio": float(r),
            "RMSE_rel": float(np.sqrt(np.mean(seg**2))) if mask.sum() else float("nan"),
            "P10_rel": float(np.percentile(seg, 10)) if mask.sum() else float("nan"),
            "P90_rel": float(np.percentile(seg, 90)) if mask.sum() else float("nan"),
        })
    return rows


def score_slip_segments(df: pd.DataFrame, edges: list[float] | None = None) -> dict:
    """Fit the table and return overall metrics plus per-bin bands."""
    params = fit_slip_segments(df, edges)
    pred = predict_slip_segments(params, df["s_fl"]).to_numpy()
    true = (df["R2"] / df["R3"]).to_numpy()
    rel_t = (pred - true) / true * 100.0
    return {
        "params": params,
        "n": int(len(df)),
        "RMSE_rel": float(np.sqrt(np.mean(rel_t**2))),
        "MAE_rel": float(np.mean(np.abs(rel_t))),
        "P10_rel": float(np.percentile(rel_t, 10)),
        "P50_rel": float(np.percentile(rel_t, 50)),
        "P90_rel": float(np.percentile(rel_t, 90)),
        "bands": _band_rows(df, params),
    }


# ---------------------------------------------------------------------------
# Grouping-constant exploration (plan.md §6; constant R2/R3 per key group)
# ---------------------------------------------------------------------------


def add_exploration_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add categorical columns used by the grouping exploration.

    - ``hp_bucket``: {<=100, 101-500, 501-1000, 1001-2000, 2000+}
    - ``slip_bin``: 10 fine bins densest below 5 % slip.
    """
    df = df.copy()
    bins_hp = [0, 100, 500, 1000, 2000, np.inf]
    labels_hp = ["<=100", "101-500", "501-1000", "1001-2000", "2000+"]
    df["hp_bucket"] = pd.cut(df["HorsePower"], bins=bins_hp, labels=labels_hp).astype(str)
    slip_edges = [0, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.0175, 0.02, 0.025, 0.03, 0.04, 1.0]
    df["slip_bin"] = pd.cut(df["s_fl"], bins=slip_edges, right=False).astype(str)
    return df


def score_grouped_constant(df: pd.DataFrame, keys: list[str], label: str) -> dict:
    """Constant geometric-mean R2/R3 per group; report group count + error."""
    keys = list(keys)
    sub = df.assign(_lr=np.log(df["R2"] / df["R3"]))
    ratio = np.exp(sub.groupby(keys, observed=True)["_lr"].transform("mean"))
    pred = ratio * df["R3"]
    rel = (pred - df["R2"]) / df["R2"]
    return {
        "label": label,
        "keys": keys,
        "n_groups": int(sub.groupby(keys, observed=True).ngroups),
        "n": int(len(df)),
        "RMSE_rel": float(np.sqrt((rel**2).mean())),
        "MAE_rel": float(rel.abs().mean()),
    }


GROUPING_CANDIDATES = {
    "voltage": ["Voltage"],
    "pole": ["PoleSpeed"],
    "pole + voltage": ["PoleSpeed", "Voltage"],
    "HP bucket": ["hp_bucket"],
    "pole + HP bucket": ["PoleSpeed", "hp_bucket"],
    "voltage + HP bucket": ["Voltage", "hp_bucket"],
    "pole + voltage + HP bucket": ["PoleSpeed", "Voltage", "hp_bucket"],
    "slip bin": ["slip_bin"],
    "pole + slip bin": ["PoleSpeed", "slip_bin"],
    "HP bucket + slip bin": ["hp_bucket", "slip_bin"],
    "pole + HP + slip bin": ["PoleSpeed", "hp_bucket", "slip_bin"],
    "pole + voltage + slip bin": ["PoleSpeed", "Voltage", "slip_bin"],
}


def grouping_score_table(df: pd.DataFrame) -> list[dict]:
    return [score_grouped_constant(add_exploration_columns(df), keys, label)
            for label, keys in GROUPING_CANDIDATES.items()]

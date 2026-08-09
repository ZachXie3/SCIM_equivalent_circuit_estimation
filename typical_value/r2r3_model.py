"""Dataset-driven R2/R3 models (plan.md Stage 2.4).

Models express the running/standstill rotor resistance ratio as an
empirical function of readily-available nameplate quantities:

    Model A:  R2/R3 = constant
    Model B:  R2/R3 = f(s_fl)
    Model C:  R2/R3 = f(s_fl, pole_count)
    Model D:  R2/R3 = f(s_fl, pole_count, HorsePower)

(Enclosure-based variants were dropped by review — the frame-size code is
an unreliable enclosure proxy.)

Each model is fitted on ``examples_eq_results.csv``, predicts R2 from R3,
and is scored with MAE, RMSE, and P10/P50/P90 relative-error bands. The
simplest model whose RMSE is within ``tolerance`` of the best is selected.

Power-law form (fitted in log-log space):

    R2/R3 = exp(b) * s_fl^a                    (Model B)
    R2/R3 = exp(b) * s_fl^a * HP^c             (Model D)
    with group-dependent (a, b) per pole count for Models C / D.

Usage:
    python r2r3_model.py                    # print scoring table
    python r2r3_report.py                   # write r2r3_report.md
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

DEFAULT_CSV = Path(__file__).resolve().parent / "examples_eq_results.csv"

MODEL_NAMES = ("A", "B", "C", "D")


def load_results(path: Path | str = DEFAULT_CSV) -> pd.DataFrame:
    return pd.read_csv(path)


def add_slip(df: pd.DataFrame) -> pd.DataFrame:
    """Add full-load slip column ``s_fl`` from nameplate values."""
    ns = 120.0 * df["Frequency"] / df["PoleSpeed"]
    return df.assign(s_fl=(ns - df["RPM"]) / ns)


def valid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Rows usable for fitting: s_fl > 0 and HP > 0."""
    return df[(df["s_fl"] > 0) & (df["HorsePower"] > 0)]


# ---------------------------------------------------------------------------
# Fitting / prediction
# ---------------------------------------------------------------------------


def fit_model_a(df: pd.DataFrame) -> dict:
    """R2/R3 = constant (geometric-mean ratio)."""
    ratio = df["R2"] / df["R3"]
    return {"const": float(np.exp(np.log(ratio).mean()))}


def predict_model_a(params: Mapping, df: pd.DataFrame) -> pd.Series:
    return pd.Series(params["const"], index=df.index) * df["R3"]


def fit_model_b(df: pd.DataFrame) -> dict:
    """R2/R3 = exp(b) * s_fl^a  (global power law on slip)."""
    x = np.log(df["s_fl"].to_numpy())
    y = np.log((df["R2"] / df["R3"]).to_numpy())
    a, b = np.polyfit(x, y, 1)
    return {"a": float(a), "b": float(b)}


def predict_model_b(params: Mapping, df: pd.DataFrame) -> pd.Series:
    ratio = np.exp(params["b"] + params["a"] * np.log(df["s_fl"].to_numpy()))
    return pd.Series(ratio, index=df.index) * df["R3"]


def fit_model_c(df: pd.DataFrame) -> dict:
    """Per-pole-count power law on slip: R2/R3 = exp(b_p) * s_fl^a_p."""
    params = {}
    for pole, g in df.groupby("PoleSpeed"):
        x = np.log(g["s_fl"].to_numpy())
        y = np.log((g["R2"] / g["R3"]).to_numpy())
        a, b = np.polyfit(x, y, 1)
        params[int(pole)] = {"a": float(a), "b": float(b)}
    return params


def predict_model_c(params: Mapping, df: pd.DataFrame) -> pd.Series:
    ratios = pd.Series(np.nan, index=df.index)
    for pole, g in df.groupby("PoleSpeed"):
        p = params.get(int(pole)) or fit_model_b(df)  # fallback for unseen pole count
        ratios.loc[g.index] = np.exp(p["b"] + p["a"] * np.log(g["s_fl"].to_numpy()))
    return ratios * df["R3"]


def fit_model_d(df: pd.DataFrame) -> dict:
    """Per-pole log-log model R2/R3 = exp(b) * s_fl^a * HP^c.

    Fitted with two predictors (log s_fl, log HP) per pole count.
    """
    params = {}
    for pole, g in df.groupby("PoleSpeed"):
        X = np.column_stack(
            [np.ones(len(g)), np.log(g["s_fl"]), np.log(g["HorsePower"])]
        )
        y = np.log((g["R2"] / g["R3"]).to_numpy())
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        params[int(pole)] = {"b": float(coef[0]), "a": float(coef[1]), "c": float(coef[2])}
    return params


def predict_model_d(params: Mapping, df: pd.DataFrame) -> pd.Series:
    ratios = pd.Series(np.nan, index=df.index)
    for pole, g in df.groupby("PoleSpeed"):
        p = params.get(int(pole))
        if p is None:
            X = np.column_stack(
                [np.ones(len(g)), np.log(g["s_fl"]), np.log(g["HorsePower"])]
            )
            y = np.log((g["R2"] / g["R3"]).to_numpy())
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            p = {"b": coef[0], "a": coef[1], "c": coef[2]}
        log_ratio = p["b"] + p["a"] * np.log(g["s_fl"]) + p["c"] * np.log(g["HorsePower"])
        ratios.loc[g.index] = np.exp(log_ratio.to_numpy())
    return ratios * df["R3"]


FIT_FUNCS = {"A": fit_model_a, "B": fit_model_b, "C": fit_model_c, "D": fit_model_d}
PREDICT_FUNCS = {
    "A": predict_model_a,
    "B": predict_model_b,
    "C": predict_model_c,
    "D": predict_model_d,
}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_model(name: str, df: pd.DataFrame) -> dict:
    """Fit model ``name`` in-sample and return fit params + error metrics.

    Metrics on predicted R2 vs dataset R2:
      - MAE / RMSE in ohms (absolute error)
      - MAE / RMSE relative error
      - P10 / P50 / P90 of relative error
    """
    params = FIT_FUNCS[name](df)
    pred = PREDICT_FUNCS[name](params, df)
    err = pred - df["R2"]
    rel = err / df["R2"]
    return {
        "name": name,
        "params": params,
        "n": int(len(df)),
        "MAE_ohm": float(err.abs().mean()),
        "RMSE_ohm": float(np.sqrt((err**2).mean())),
        "MAE_rel": float(rel.abs().mean()),
        "RMSE_rel": float(np.sqrt((rel**2).mean())),
        "P10_rel": float(rel.quantile(0.10)),
        "P50_rel": float(rel.quantile(0.50)),
        "P90_rel": float(rel.quantile(0.90)),
    }


def select_model(scores: Mapping[str, dict], tolerance: float = 0.05) -> str:
    """Simplest model whose RMSE_rel is within ``tolerance`` of the best."""
    best_rmse = min(scores[name]["RMSE_rel"] for name in MODEL_NAMES)
    for name in MODEL_NAMES:
        if scores[name]["RMSE_rel"] <= best_rmse * (1.0 + tolerance):
            return name
    return min(scores, key=lambda n: scores[n]["RMSE_rel"])


def score_table(df: pd.DataFrame) -> tuple[list[dict], str]:
    """Fit all models, return (rows for a report table, selected model name)."""
    rows = [score_model(name, df) for name in MODEL_NAMES]
    selected = select_model({r["name"]: r for r in rows})
    return rows, selected


# ---------------------------------------------------------------------------
# Grouping-constant exploration (group R2/R3 by categorical keys, constant per group)
# ---------------------------------------------------------------------------


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


def add_exploration_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add the categorical columns used by the grouping exploration.

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


def grouping_score_table(df: pd.DataFrame) -> list[dict]:
    return [score_grouped_constant(add_exploration_columns(df), keys, label)
            for label, keys in GROUPING_CANDIDATES.items()]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    df = add_slip(load_results())
    df = valid_rows(df)
    rows, selected = score_table(df)

    print(f"Rows used for fitting: {len(df)} (invalid slip / HP excluded)")
    print()
    header = (f"{'Model':<6}{'MAE(ohm)':>10}{'RMSE(ohm)':>10}{'MAE(%)':>9}"
              f"{'RMSE(%)':>9}{'P10(%)':>9}{'P50(%)':>9}{'P90(%)':>9}")
    print(header)
    print("-" * len(header))
    for r in rows:
        mark = "  <-- selected" if r["name"] == selected else ""
        print(
            f"{r['name']:<6}"
            f"{r['MAE_ohm']:>10.4f}"
            f"{r['RMSE_ohm']:>10.4f}"
            f"{r['MAE_rel']*100:>9.2f}"
            f"{r['RMSE_rel']*100:>9.2f}"
            f"{r['P10_rel']*100:>9.2f}"
            f"{r['P50_rel']*100:>9.2f}"
            f"{r['P90_rel']*100:>9.2f}"
            f"{mark}"
        )


if __name__ == "__main__":
    main()
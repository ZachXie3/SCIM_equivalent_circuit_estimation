"""Task 2.4 — Tests for the dataset-driven R2/R3 analysis.

Covers the ``r2r3`` package:
- ``r2r3.models``: Model A-D fitting/prediction, scoring, model selection.
- ``r2r3.constant_bin``: the approved constant-per-bin (slip-segment) method,
  plus the grouping-constant exploration.

Run from the repo root (or with pytest from ``typical_value/``):
    python -m pytest typical_value/tests/test_r2r3_model.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_TV = Path(__file__).resolve().parent.parent
_ROOT = _TV.parent
sys.path.insert(0, str(_TV))
sys.path.insert(0, str(_ROOT))

from typical_value.r2r3 import constant_bin as cb  # noqa: E402
from typical_value.r2r3 import models as m  # noqa: E402
from typical_value.r2r3 import data as data_mod  # noqa: E402

RESULTS_CSV = _TV / "data" / "eq_parameters.csv"


@pytest.fixture(scope="module")
def df():
    return data_mod.valid_rows(data_mod.add_slip(data_mod.load_results(RESULTS_CSV)))


def test_slip_computed():
    df = data_mod.load_results(RESULTS_CSV)
    out = data_mod.add_slip(df)
    assert "s_fl" in out.columns
    ns = 120.0 * df["Frequency"] / df["PoleSpeed"]
    assert np.allclose(out["s_fl"], (ns - df["RPM"]) / ns)


def test_valid_rows_excludes_invalid():
    df = data_mod.add_slip(data_mod.load_results(RESULTS_CSV))
    out = data_mod.valid_rows(df)
    assert len(out) < len(df)
    assert (out["s_fl"] > 0).all()
    assert (out["HorsePower"] > 0).all()


def test_fit_and_predict_all_models(df):
    for name in m.MODEL_NAMES:
        params = m.FIT_FUNCS[name](df)
        pred = m.PREDICT_FUNCS[name](params, df)
        assert len(pred) == len(df)
        assert np.isfinite(pred).all()
        assert (pred > 0).all()


def test_model_a_is_constant_ratio(df):
    params = m.fit_model_a(df)
    ratio = df["R2"] / df["R3"]
    expected = np.exp(np.log(ratio).mean())
    assert params["const"] == pytest.approx(expected)
    pred = m.predict_model_a(params, df)
    assert np.allclose(pred / df["R3"], params["const"])


def test_predict_handles_filtered_index():
    df = data_mod.add_slip(data_mod.load_results(RESULTS_CSV))
    df = data_mod.valid_rows(df)
    params = m.fit_model_d(df)
    pred = m.predict_model_d(params, df)
    assert pred.index.equals(df.index)


def test_model_d_reproduces_perfect_linear_fit():
    """Model D on synthetic data with exact power-law must fit perfectly."""
    rng = np.random.default_rng(42)
    n = 200
    poles = rng.integers(2, 10, size=n) * 2  # 2,4,6,8
    s = np.exp(rng.uniform(np.log(0.005), np.log(0.08), size=n))
    hp = np.exp(rng.uniform(np.log(1), np.log(2000), size=n))
    r3 = np.exp(rng.uniform(np.log(0.01), np.log(5), size=n))
    # exact R2/R3 = exp(b + a*log(s) + c*log(hp)), same a,b,c for all poles
    a, b, c = 1.0, 2.5, -0.3
    ratio = np.exp(b + a * np.log(s) + c * np.log(hp))
    r2 = ratio * r3
    df = pd.DataFrame(
        {"PoleSpeed": poles, "s_fl": s, "HorsePower": hp, "R2": r2, "R3": r3}
    )
    params = m.fit_model_d(df)
    pred = m.predict_model_d(params, df)
    assert np.allclose(pred / df["R2"], 1.0, rtol=1e-6)


def test_score_metrics(df):
    for name in m.MODEL_NAMES:
        r = m.score_model(name, df)
        assert r["name"] == name
        assert r["n"] == len(df)
        assert r["RMSE_ohm"] >= 0
        assert 0 <= r["MAE_rel"] <= 2
        assert r["P10_rel"] <= r["P50_rel"] <= r["P90_rel"]


def test_predict_and_base_deliver_finite(df):
    """Across pole groups present in the data, D prediction stays finite."""
    params = m.fit_model_d(df)
    for pole in df["PoleSpeed"].unique():
        g = df[df["PoleSpeed"] == pole]
        pred = m.predict_model_d({int(pole): params[int(pole)]}, g)
        assert np.isfinite(pred).all()


def test_selection_returns_valid_name(df):
    scores = {name: m.score_model(name, df) for name in m.MODEL_NAMES}
    selected = m.select_model(scores)
    assert selected in m.MODEL_NAMES


def test_select_prefers_simpler_within_tolerance():
    scores = {
        "A": {"RMSE_rel": 0.50},
        "B": {"RMSE_rel": 0.51},
        "C": {"RMSE_rel": 0.53},
        "D": {"RMSE_rel": 0.10},
    }
    assert m.select_model(scores) == "D"  # best is far better
    scores2 = {
        "A": {"RMSE_rel": 0.50},
        "B": {"RMSE_rel": 0.51},
        "C": {"RMSE_rel": 0.53},
        "D": {"RMSE_rel": 0.52},
    }
    assert m.select_model(scores2) == "A"  # simplest within 5% of best


# ---------------------------------------------------------------------------
# constant_bin — approved constant-per-bin (slip-segment) method
# ---------------------------------------------------------------------------


def test_fit_slip_segments_extrapolates_top_band(df):
    params = cb.fit_slip_segments(df)
    assert "edges" in params and "ratios" in params
    top = [k for k in params["ratios"] if k.left >= cb.EXTRAPOLATE_AT]
    assert len(top) == 1
    assert top[0].right >= 1.0
    assert params["ratios"][top[0]] == 1.0
    assert all(0.0 < v <= 1.0 for v in params["ratios"].values())


def test_predict_slip_segments_matches_band(df):
    params = cb.fit_slip_segments(df)
    pred = cb.predict_slip_segments(params, df["s_fl"])
    assert len(pred) == len(df)
    assert np.isfinite(pred).all()
    assert (pred >= 0).all()
    # every fitted row maps to its band's geometric-mean ratio
    cats = pd.cut(df["s_fl"], params["edges"], right=False)
    for k, r in params["ratios"].items():
        mask = (cats == k).to_numpy()
        assert np.allclose(pred[mask], r)


def test_predict_slip_segments_out_of_range_clamps(df):
    params = cb.fit_slip_segments(df)
    pred = cb.predict_slip_segments(params, [1.5])  # above the last edge
    assert pred.iloc[0] == pytest.approx(1.0)


def test_predict_slip_segments_r2(df):
    params = cb.fit_slip_segments(df)
    r2 = cb.predict_slip_segments_r2(params, df)
    ratio = cb.predict_slip_segments(params, df["s_fl"])
    assert np.allclose(r2, ratio * df["R3"])
    assert (r2 > 0).all()
    assert (r2 <= df["R3"]).all()


def test_score_slip_segments_reports_expected_accuracy(df):
    """Overall error of the table method matches the report (§11.1): ~34%."""
    sc = cb.score_slip_segments(df)
    assert sc["n"] == len(df)
    assert sc["RMSE_rel"] == pytest.approx(34.0, abs=1.0)
    assert 0 < sc["MAE_rel"] < sc["RMSE_rel"]
    assert sc["P10_rel"] <= sc["P50_rel"] <= sc["P90_rel"]
    assert len(sc["bands"]) == len(sc["params"]["ratios"])
    for band in sc["bands"]:
        assert band["n"] >= 1
        assert 0.0 < band["ratio"] <= 1.0


def test_grouping_score_table_runs(df):
    table = cb.grouping_score_table(df)
    assert len(table) == len(cb.GROUPING_CANDIDATES)
    for g in table:
        assert g["n"] == len(df)
        assert g["n_groups"] >= 1
        assert 0 < g["RMSE_rel"] < 1.5

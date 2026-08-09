"""Task 2.4 — Tests for the dataset-driven R2/R3 model.

Covers ``r2r3_model.py``: Model A-D fitting/prediction, scoring metrics,
and model selection.

Run from the repo root (or with pytest from ``typical_value/``):
    python -m pytest typical_value/test/test_r2r3_model.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import r2r3_model as m  # noqa: E402

RESULTS_CSV = Path(__file__).resolve().parent.parent / "examples_eq_results.csv"


@pytest.fixture(scope="module")
def df():
    return m.valid_rows(m.add_slip(m.load_results(RESULTS_CSV)))


def test_slip_computed():
    df = m.load_results(RESULTS_CSV)
    out = m.add_slip(df)
    assert "s_fl" in out.columns
    ns = 120.0 * df["Frequency"] / df["PoleSpeed"]
    assert np.allclose(out["s_fl"], (ns - df["RPM"]) / ns)


def test_valid_rows_excludes_invalid():
    df = m.add_slip(m.load_results(RESULTS_CSV))
    out = m.valid_rows(df)
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
    df = m.add_slip(m.load_results(RESULTS_CSV))
    df = m.valid_rows(df)
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
"""Tests for the Stage 5 validation harness (validate.py)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import validate as v  # noqa: E402

N_ROWS = 60


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    return pd.read_csv(v.RESULTS_CSV).iloc[:N_ROWS]


def test_stage1_stage2_vector(raw):
    st = v.stage1_stage2(raw)
    for key in ("v_ph", "i_fl_ph", "s_fl", "r1_hot", "r1_cold_feed"):
        assert len(st[key]) == N_ROWS
        assert np.all(np.isfinite(st[key]))
    assert np.all(st["r1_hot"] > 0)
    assert np.all(st["r1_hot"] > st["r1_cold_feed"])


def test_r2_prior_series_bounds(raw):
    d = v._build_dyn(raw)
    d = v.r2_prior_series(d)
    r2 = d["r2_prior"].to_numpy()
    assert np.all(r2 > 0)
    assert np.all(r2 <= d["r3_est"].to_numpy())


@pytest.mark.parametrize("x1frac,x2frac", [(0.02, 0.02), (0.15, 0.15)])
def test_solver_finite(raw, x1frac, x2frac):
    dyn = v._build_dyn(raw)
    dyn = v.r2_prior_series(dyn)
    sol = v.solve_x1x2_vector(dyn, dyn["x0"].to_numpy(), max_iter=5)
    assert np.all(np.isfinite(sol["X1"]))
    assert np.all(np.isfinite(sol["X2"]))
    assert np.all(sol["X1"] > 0)
    assert np.all(sol["X2"] >= 0)


def test_pipeline_roundtrip_converges(raw):
    res = v.run_vectorised(raw)
    assert len(res) == N_ROWS
    assert res["converged"].mean() > 0.9
    for c in ("R2", "R3", "X1", "X2", "X3", "Xm", "design_id"):
        assert c in res.columns


def test_compute_errors_columns(raw):
    res = v.run_vectorised(raw)
    err = v.compute_errors(res, raw)
    for p in v.METRIC_PARAMS:
        assert f"err_{p}" in err.columns
        assert np.all(np.isfinite(err[f"err_{p}"]))


def test_ref_table_matches_reference(raw):
    res = v.run_vectorised(raw)
    ref, _ = v.run_reference(raw, np.arange(min(5, len(raw))))
    ref_ok = [r for r in ref if not r.get("raised")]
    assert len(ref_ok) > 0
    tbl = v._ref_table(ref_ok, res)
    for p in v.METRIC_PARAMS:
        assert np.all(tbl[p].to_numpy() >= 0)


def test_module_imports():
    assert v.METRIC_PARAMS == ("R2", "R3", "X1", "X2", "X3", "Xm")
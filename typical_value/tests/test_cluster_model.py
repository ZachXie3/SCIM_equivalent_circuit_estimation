"""Tests for the weighted-cluster R2/R3 analysis (cluster-plan.md).

Covers ``typical_value/cluster/cluster_model.py`` (data prep, encoding, the
weighted soft-clustering fit, hard lookup, predict, save/load) and the split
helpers in ``cluster_analysis.py``.

Run from the repo root (or with pytest from ``typical_value/``):
    python -m pytest typical_value/tests/test_cluster_model.py -q
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_TV = Path(__file__).resolve().parent.parent
_ROOT = _TV.parent
sys.path.insert(0, str(_TV))
sys.path.insert(0, str(_ROOT))

from typical_value.cluster import cluster_analysis as ca  # noqa: E402
from typical_value.cluster import cluster_model as cm  # noqa: E402

DATA_CSV = _TV / "data" / "eq_parameters.csv"
RATIO_MAX = cm.RATIO_MAX


@pytest.fixture(scope="module")
def df():
    return cm.prepare(pd.read_csv(DATA_CSV))


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


def test_prepare_drops_invalid_rows():
    df = cm.prepare(pd.read_csv(DATA_CSV))
    assert len(df) == 3853
    assert (df["slip"] > 0).all()
    assert (df["ratio"] < RATIO_MAX).all()
    assert df["y"].isna().sum() == 0


def test_prepare_adds_derived_columns(df):
    for c in ("slip", "ratio", "y", "nema_design_o", "nema_kva_o"):
        assert c in df.columns


def test_nema_design_ordinal_mapping(df):
    assert set(df["nema_design_o"].unique()).issubset({0, 1, 2, 3})
    assert df.loc[df["NemaDesign"].isna(), "nema_design_o"].eq(0).all()


def test_nema_kva_ordinal_mapping(df):
    assert set(df["nema_kva_o"].unique()).issubset(range(14))
    assert df["nema_kva_o"].isna().sum() == 0


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


def test_encode_full_shape(df):
    scaler = cm.fit_scaler(df)
    X, slices = cm.encode(df, scaler)
    assert X.shape == (len(df), 12)
    assert len(slices) == 7
    assert slices[-1] == (8, 12)
    assert np.isfinite(X).all()


def test_encode_subset_ablation(df):
    scaler = cm.fit_scaler(df)
    X, slices = cm.encode(df, scaler, [1, 2, 3, 4, 5, 6])  # drop slip
    assert X.shape == (len(df), 11)
    assert len(slices) == 6


def test_fit_scaler_standardizes(df):
    scaler = cm.fit_scaler(df)
    X, slices = cm.encode(df, scaler)
    assert np.allclose(X[:, 0].mean(), 0.0, atol=1e-10)
    assert np.allclose(X[:, 0].std(), 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_split(df):
    d = df.iloc[:700]
    dtr, dva, dte = d.iloc[:450], d.iloc[450:600], d.iloc[600:]
    scaler = cm.fit_scaler(dtr)
    Xtr, sl = cm.encode(dtr, scaler)
    Xva, _ = cm.encode(dva, scaler)
    Xte, _ = cm.encode(dte, scaler)
    return {
        "dtr": dtr, "dva": dva, "dte": dte, "scaler": scaler, "slices": sl,
        "Xtr": Xtr, "Xva": Xva, "Xte": Xte,
        "ytr": dtr["y"].to_numpy(), "yva": dva["y"].to_numpy(),
        "yte": dte["y"].to_numpy(),
    }


def test_fit_weights_nonneg(small_split):
    m = cm.WeightedClusterModel(n_clusters=6, max_steps=30, n_restarts=1, seed=7)
    m.fit(small_split["Xtr"], small_split["ytr"], small_split["slices"],
          small_split["Xva"], small_split["yva"])
    assert m.w.shape == (7,)
    assert (m.w >= 0).all() and np.isfinite(m.w).all()


def test_penalty_prunes_weak_features(small_split):
    """A strong count penalty drives some weights to ~0 (inactive features)."""
    m = cm.WeightedClusterModel(n_clusters=6, max_steps=300, n_restarts=1,
                                seed=7, beta=5.0)
    m.fit(small_split["Xtr"], small_split["ytr"], small_split["slices"])
    active = m.active_features
    assert len(active) < 7
    assert all(m.w[i] <= cm.WeightedClusterModel.ACTIVE_W
               for i in range(7) if i not in active)


def test_fit_deterministic(small_split):
    kw = dict(n_clusters=6, max_steps=30, n_restarts=1, seed=7)
    m1 = cm.WeightedClusterModel(**kw)
    m1.fit(small_split["Xtr"], small_split["ytr"], small_split["slices"])
    m2 = cm.WeightedClusterModel(**kw)
    m2.fit(small_split["Xtr"], small_split["ytr"], small_split["slices"])
    assert np.allclose(m1.w, m2.w)


def test_build_lookup_and_predict(small_split):
    s = small_split
    m = cm.WeightedClusterModel(n_clusters=6, max_steps=30, n_restarts=1, seed=7)
    m.fit(s["Xtr"], s["ytr"], s["slices"])
    df_fit = pd.concat([s["dtr"], s["dva"]])
    m.build_lookup(df_fit, np.vstack([s["Xtr"], s["Xva"]]),
                   np.concatenate([s["ytr"], s["yva"]]), s["slices"], s["scaler"])
    pred = m.predict_ratio(s["Xte"])
    assert pred.shape == (len(s["dte"]),)
    assert (pred > 0).all() and (pred <= RATIO_MAX).all()
    assert np.isfinite(pred).all()
    assert m.n_hard.sum() == len(df_fit)


def test_assign_returns_cluster_ids(small_split):
    s = small_split
    m = cm.WeightedClusterModel(n_clusters=6, max_steps=30, n_restarts=1, seed=7)
    m.fit(s["Xtr"], s["ytr"], s["slices"])
    df_fit = pd.concat([s["dtr"], s["dva"]])
    m.build_lookup(df_fit, np.vstack([s["Xtr"], s["Xva"]]),
                   np.concatenate([s["ytr"], s["yva"]]), s["slices"], s["scaler"])
    labels = m.assign(s["Xte"])
    assert labels.shape == (len(s["dte"]),)
    assert (labels >= 0).all() and (labels < m.n_clusters).all()


def test_save_load_roundtrip(small_split, tmp_path):
    s = small_split
    m = cm.WeightedClusterModel(n_clusters=6, max_steps=30, n_restarts=1, seed=7)
    m.fit(s["Xtr"], s["ytr"], s["slices"])
    df_fit = pd.concat([s["dtr"], s["dva"]])
    m.build_lookup(df_fit, np.vstack([s["Xtr"], s["Xva"]]),
                   np.concatenate([s["ytr"], s["yva"]]), s["slices"], s["scaler"])
    pred = m.predict_ratio(s["Xte"])

    npz = tmp_path / "model.npz"
    cm.save_lookup(m, s["scaler"], s["slices"], npz)
    params = cm.load_lookup(npz)
    pred2 = cm.predict_lookup(params, s["dte"]).to_numpy()
    assert np.allclose(pred, pred2, atol=1e-12)
    assert params["n_clusters"] == 6
    assert len(params["scaler"]) == 5
    assert params["slices"] == s["slices"]


def test_weighted_kmeans_empty_guard(small_split):
    mu, labels = cm.weighted_kmeans(small_split["Xtr"], small_split["slices"],
                                    np.full(7, 0.5), 4, seed=3, max_iter=5)
    assert mu.shape == (4, 12)
    assert labels.shape == (len(small_split["Xtr"]),)
    assert set(np.unique(labels)).issubset(set(range(4)))


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------


def test_stratified_split_sizes(df):
    tr, va, te = ca.stratified_split(df, seed=42)
    assert len(tr) + len(va) + len(te) == len(df)
    assert len(tr) > len(va) > 0 and len(te) > 0


def test_stratified_split_keeps_ratio_range(df):
    tr, _, te = ca.stratified_split(df, seed=42)
    assert tr["ratio"].min() > 0 and tr["ratio"].max() < RATIO_MAX
    assert te["ratio"].min() > 0 and te["ratio"].max() < RATIO_MAX

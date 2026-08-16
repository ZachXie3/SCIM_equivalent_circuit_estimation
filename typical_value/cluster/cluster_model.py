"""Weighted soft-clustering R2/R3 model (see ``cluster-plan.md``).

Learns per-feature weights ``w`` for a K-means-style clustering of the
nameplate features, so that each cluster's mean log-ratio ``y = ln(R2/R3)``
predicts the target. Only the feature weights are learned (the NN); cluster
centroids ``mu`` and means ``m`` are recomputed deterministically from the
soft assignments each step (EM-style).

    w_g   = softmax(z)_g * G                per-feature weight, sum(w) = G
    d_ic  = sum_g w_g * ||x_{i,g} - mu_{c,g}||^2    weighted squared distance
    a_ic  = softmax(-d_ic / tau)            soft assignment
    y_hat = sum_c a_ic * m_c                prediction (log space)
    L     = RMSE(y - y_hat) + beta * count(active) * rmse_ref

Feature-count penalty (active = weight not 0): ``beta`` is the cost of one
active feature as a fraction of the reference RMSE, so a feature is retained
only if removing it costs more than ``beta`` pp of RMSE. Weights are
normalised (``sum(w) = G``) so the penalty cannot be gamed by shrinking all
weights uniformly — the only way to reduce the count is to zero a feature.
The delivered model is the hard-assignment lookup built on the active
features (``cluster_lookup.csv``).

Usage (programmatic):
    from typical_value.cluster import cluster_model as cm
    df   = cm.prepare(pd.read_csv(DATA_CSV))
    scal = cm.fit_scaler(df_train)                 # train split only
    X, groups = cm.encode(df, scal)
    model = cm.WeightedClusterModel().fit(Xtr, ytr, groups, Xva, yva)
    model.build_lookup(df_tr, Xtr, ytr, groups, scal)
    pred = model.predict_ratio(Xte)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

try:
    from scipy.cluster.vq import kmeans2
    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    kmeans2 = None
    _HAS_SCIPY = False

EPS = 1e-12
RATIO_MAX = 1.0  # physical bound R2 <= R3

# ---------------------------------------------------------------------------
# Feature spec (cluster-plan.md section 2.1)
# ---------------------------------------------------------------------------

# Feature groups in order; each group has one learned weight.
#   kind:
#     "numeric"  -> z(x)                        (1 dim)
#     "log"      -> z(ln x)                     (1 dim)
#     "ordinal"  -> z(rank)                     (1 dim)
#     "onehot"   -> one-hot dummies             (n_levels dims)
FEATURE_GROUPS = [
    {"name": "slip",        "kind": "numeric", "source": "slip"},
    {"name": "HorsePower",  "kind": "log",     "source": "HorsePower"},
    {"name": "PoleSpeed",   "kind": "numeric", "source": "PoleSpeed"},
    {"name": "NemaDesign",  "kind": "ordinal", "source": "nema_design_o"},
    {"name": "NemaKVACode", "kind": "ordinal", "source": "nema_kva_o"},
    {"name": "BarMaterial", "kind": "onehot",  "source": "BarMaterial",
     "levels": ["AL", "CU", "Special"]},
    {"name": "Name",        "kind": "onehot",  "source": "Name",
     "levels": ["NEMA Premium4 Brand", "Standard / High Efficiency",
                "TIC EPACT", "TIC Premium Brand"]},
]
GROUP_NAMES = [g["name"] for g in FEATURE_GROUPS]
N_GROUPS = len(GROUP_NAMES)

# NemaDesign ordinal: none=0, A=1, B=2, C=3 (cluster-plan.md section 2.2).
NEMA_DESIGN_MAP = {"A": 1, "A,B": 2, "A,B,C": 3, "C": 3}

# NemaKVACode ordinal: one level per present letter, alphabetical (plan 2.3).
NEMA_KVA_RANK = {
    "C": 0, "D": 1, "E": 2, "F": 3, "G": 4, "H": 5, "J": 6, "K": 7,
    "L": 8, "M": 9, "N": 10, "P": 11, "R": 12, "V": 13,
}
NEMA_KVA_MODAL = 4  # missing -> modal letter G


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Drop invalid rows and add derived columns (slip, ratio, y, ordinals).

    Drops (cluster-plan.md section 2): ``slip <= 0`` and ``R2/R3 >= 1.0``.
    Adds ``slip``, ``ratio``, ``y = ln(ratio)``, ``nema_design_o``,
    ``nema_kva_o``.
    """
    df = df.copy()
    df["slip"] = (df["SRPM"] - df["RPM"]) / df["SRPM"]
    df["ratio"] = df["R2"] / df["R3"]
    df["y"] = np.log(df["ratio"])
    df["nema_design_o"] = df["NemaDesign"].map(NEMA_DESIGN_MAP).fillna(0).astype(int)
    df["nema_kva_o"] = df["NemaKVACode"].map(NEMA_KVA_RANK).fillna(NEMA_KVA_MODAL).astype(int)
    df = df[(df["slip"] > 0) & (df["ratio"] < RATIO_MAX)]
    return df.reset_index(drop=True)


def fit_scaler(df: pd.DataFrame) -> dict:
    """Per-group (mean, std) for the numeric/log/ordinal groups, from ``df``.

    Fit on the **train** split only and reused for validation/test.
    """
    scaler = {}
    for g in FEATURE_GROUPS:
        if g["kind"] in ("numeric", "log", "ordinal"):
            v = df[g["source"]].to_numpy(dtype=float)
            if g["kind"] == "log":
                v = np.log(np.maximum(v, EPS))
            scaler[g["name"]] = (float(v.mean()), float(max(v.std(), EPS)))
    return scaler


def encode(
    df: pd.DataFrame,
    scaler: dict,
    group_indices: list[int] | None = None,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Encode ``df`` into the design matrix; return (X, group slices).

    ``group_indices`` selects a subset of ``FEATURE_GROUPS`` (used by the
    feature-ablation and pruning runs); the default is all groups.
    """
    indices = list(range(len(FEATURE_GROUPS))) if group_indices is None else list(group_indices)
    parts: list[np.ndarray] = []
    slices: list[tuple[int, int]] = []
    start = 0
    for gi in indices:
        g = FEATURE_GROUPS[gi]
        if g["kind"] in ("numeric", "log", "ordinal"):
            v = df[g["source"]].to_numpy(dtype=float)
            if g["kind"] == "log":
                v = np.log(np.maximum(v, EPS))
            mean, std = scaler[g["name"]]
            col = (v - mean) / std
            parts.append(col.reshape(-1, 1))
        else:  # onehot
            dummies = pd.get_dummies(df[g["source"]].astype(str), dtype=float) \
                .reindex(columns=g["levels"], fill_value=0.0)
            parts.append(dummies.to_numpy())
        end = start + parts[-1].shape[1]
        slices.append((start, end))
        start = end
    X = np.hstack(parts).astype(float)
    return X, slices


# ---------------------------------------------------------------------------
# Distance / assignment helpers
# ---------------------------------------------------------------------------


def group_sq_dist(X: np.ndarray, slices: list[tuple[int, int]], mu: np.ndarray) -> np.ndarray:
    """Per-group squared centroid distances, shape (N, C, G)."""
    n, c = len(X), len(mu)
    s = np.empty((n, c, len(slices)))
    for g, (a, b) in enumerate(slices):
        d = X[:, None, a:b] - mu[None, :, a:b]
        s[:, :, g] = np.einsum("ijk,ijk->ij", d, d)
    return s


def softmax_rows(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def em_recompute(
    X: np.ndarray, y: np.ndarray, slices: list[tuple[int, int]], a: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted centroids and cluster log-ratio means from soft assignments."""
    denom = a.sum(axis=0)
    denom = np.maximum(denom, EPS)
    mu = (a.T @ X) / denom[:, None]
    m = (a.T @ y) / denom
    return mu, m


def hard_assign(X: np.ndarray, slices: list[tuple[int, int]], w: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Nearest weighted centroid per row (hard assignment)."""
    s = group_sq_dist(X, slices, mu)
    d = np.einsum("icg,g->ic", s, w)
    return np.argmin(d, axis=1)


# ---------------------------------------------------------------------------
# Weighted hard K-means (for the delivered lookup + unweighted baseline)
# ---------------------------------------------------------------------------


def weighted_kmeans(
    X: np.ndarray,
    slices: list[tuple[int, int]],
    w: np.ndarray,
    n_clusters: int,
    seed: int = 42,
    max_iter: int = 200,
    init_mu: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Lloyd's algorithm with per-feature weighted distance."""
    rng = np.random.default_rng(seed)
    if init_mu is not None:
        mu = np.asarray(init_mu, dtype=float).copy()
    else:
        perm = rng.permutation(len(X))[:n_clusters]
        mu = X[perm].copy()
        if _HAS_SCIPY:
            mu, _ = kmeans2(X, mu, minit="matrix", iter=10, seed=seed)
    for _ in range(max_iter):
        labels = hard_assign(X, slices, w, mu)
        new_mu = np.empty_like(mu)
        for c in range(n_clusters):
            sel = labels == c
            new_mu[c] = X[sel].mean(axis=0) if sel.sum() else mu[c]
        if np.allclose(new_mu, mu, atol=1e-12, rtol=0):
            mu = new_mu
            break
        mu = new_mu
    labels = hard_assign(X, slices, w, mu)
    return mu, labels


# ---------------------------------------------------------------------------
# Adam
# ---------------------------------------------------------------------------


def adam_update(
    z: np.ndarray,
    g: np.ndarray,
    state: dict,
    t: int,
    lr: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
) -> np.ndarray:
    m_ = state["m"]; v = state["v"]
    m_ = beta1 * m_ + (1 - beta1) * g
    v = beta2 * v + (1 - beta2) * g * g
    mh = m_ / (1 - beta1 ** t)
    vh = v / (1 - beta2 ** t)
    state["m"] = m_; state["v"] = v
    return z - lr * mh / (np.sqrt(vh) + 1e-8)


# ---------------------------------------------------------------------------
# Weighted soft-cluster model
# ---------------------------------------------------------------------------


@dataclass
class WeightedClusterModel:
    """Learns per-feature weights for a K-means-style R2/R3 lookup.

    Penalty: ``beta * count_active * rmse_ref``, where ``count_active`` is a
    smooth surrogate for the number of active (non-zero-weight) features and
    ``rmse_ref`` is a fixed reference RMSE. ``beta`` is thus interpretable as
    "the cost, in RMSE percentage points, of keeping one feature" - a feature
    is retained only if removing it would cost more than that. (plan 3.3)

    Weights use ``w = softmax(z) * G`` (so ``sum(w) = G``; uniform shrinking is
    impossible, which keeps the count penalty honest - the only way to reduce
    the count is to zero a feature). A feature is **active** when
    ``w > ACTIVE_W``.
    """

    n_clusters: int = 6
    tau: float = 0.5          # fixed temperature (annealing is future work)
    beta: float = 0.1         # cost per active feature, as fraction of rmse_ref
    rmse_ref: float | None = None  # fixed RMSE reference (baseline); set in fit
    lr: float = 1e-2
    max_steps: int = 500
    seed: int = 42
    n_restarts: int = 3

    w: np.ndarray = field(default=None, repr=False)          # (G,)
    mu: np.ndarray = field(default=None, repr=False)         # (C, D) learned (soft)
    m: np.ndarray = field(default=None, repr=False)          # (C,) soft log-ratio means
    mu_hard: np.ndarray = field(default=None, repr=False)    # (C, D) hard lookup centroids
    m_hard: np.ndarray = field(default=None, repr=False)     # (C,) hard log-ratio means
    n_hard: np.ndarray = field(default=None, repr=False)     # (C,) cluster counts
    lookup_rmse: float | None = field(default=None, repr=False)
    _slices: list | None = field(default=None, repr=False)
    _active: np.ndarray | None = field(default=None, repr=False)  # bool per group

    # A feature with w above this is "active" (weight effectively non-zero).
    ACTIVE_W = 0.05
    # Soft-count curvature: count ~ 1 - exp(-K*w). Small K keeps a useful
    # gradient across the whole weight range (so the penalty can push
    # borderline features to ~0); with sum(w) = G a kept feature is ~1-5
    # (count ~ 1) and a pruned one is ~0.
    COUNT_K = 3.0

    @staticmethod
    def _weight(z: np.ndarray) -> np.ndarray:
        """softmax(z) * len(z)  ->  non-negative, sum(w) = len(z)."""
        z = z - z.max()
        e = np.exp(z)
        s = e / e.sum()
        return s * len(z)

    @staticmethod
    def _count_surrogate(w: np.ndarray) -> np.ndarray:
        """Smooth, differentiable count of non-zero weights (0 at w=0, ~1 for w~1)."""
        return 1.0 - np.exp(-WeightedClusterModel.COUNT_K * np.maximum(w, 0.0))

    def _loss_and_grad(
        self, X: np.ndarray, y: np.ndarray, slices: list[tuple[int, int]], z: np.ndarray
    ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """Forward + gradient of z for one Adam step (mu/m recomputed EM-style)."""
        n = len(X)
        w = self._weight(z)
        mu = self.mu if self.mu is not None else np.zeros((self.n_clusters, X.shape[1]))
        m = self.m if self.m is not None else np.zeros(self.n_clusters)

        s = group_sq_dist(X, slices, mu)
        a = softmax_rows(-np.einsum("icg,g->ic", s, w) / max(self.tau, EPS))
        mu, m = em_recompute(X, y, slices, a)
        self.mu, self.m = mu, m

        s = group_sq_dist(X, slices, mu)
        d = np.einsum("icg,g->ic", s, w)
        a = softmax_rows(-d / max(self.tau, EPS))
        yhat = a @ m

        rmse = float(math.sqrt(np.mean((y - yhat) ** 2)))
        count = self._count_surrogate(w)
        penalty = self.beta * float(count.sum()) * (self.rmse_ref or rmse)
        loss = rmse + penalty

        # gradients (mu, m detached - recomputed, treated as constants here)
        g_y = -2.0 * (y - yhat) / n                            # (N,)
        # softmax chain: da_ic/dz_ij = a_ic (delta_cj - a_ij), z = -d/tau,
        # so dL/dd_ij = -(1/tau) * a_ij * g_y_i * (m_j - yhat_i)
        dL_dd = -(1.0 / self.tau) * a * g_y[:, None] * (m[None, :] - yhat[:, None])  # (N,C)
        dL_dw = np.einsum("ic,icg->g", dL_dd, s)
        # penalty grad: beta * rmse_ref * d(count)/dw
        dcount_dw = self.COUNT_K * np.exp(-self.COUNT_K * np.maximum(w, 0.0))
        dL_dw = dL_dw + self.beta * (self.rmse_ref or rmse) * dcount_dw
        # w = softmax(z) * G, G = len(w):
        #   dw_g/dz_j = w_g * (delta_gj - w_j/G)
        # so  dL/dz_j = dL/dw_j*w_j - (w_j/G) * sum_g(dL/dw_g * w_g)
        g_ = len(w)
        dL_dz = dL_dw * w - (w / g_) * np.sum(dL_dw * w)
        return loss, dL_dz, mu, m

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        slices: list[tuple[int, int]],
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "WeightedClusterModel":
        """Learn the feature weights ``z`` by Adam on the soft-cluster loss.

        Returns ``self`` with ``w`` set to the best (validation) weights.
        """
        if not _HAS_SCIPY:
            raise RuntimeError("cluster model init requires scipy.cluster.vq.kmeans2")

        self._slices = list(slices)
        best = None
        n_feat = len(slices)

        # Fixed RMSE reference so the count penalty is scale-stable: use the
        # RMSE of a constant predictor (predicting the mean log-ratio).
        if self.rmse_ref is None:
            self.rmse_ref = float(math.sqrt(np.mean((y - y.mean()) ** 2)))

        for r in range(self.n_restarts):
            seed = self.seed + r
            # even weight init: w = 1 for all features -> z = 0 (uniform softmax)
            z = np.zeros(n_feat)
            # init centroids with even weights
            mu, _ = kmeans2(X, self.n_clusters, minit="++", iter=10, seed=seed)
            self.mu = mu
            self.m = np.zeros(self.n_clusters)
            state = {"m": np.zeros_like(z), "v": np.zeros_like(z)}

            run_best = None
            for t in range(1, self.max_steps + 1):
                loss, g, mu, m = self._loss_and_grad(X, y, slices, z)
                z = adam_update(z, g, state, t, self.lr)
                w = self._weight(z)

                if X_val is not None and y_val is not None:
                    # score on validation using current soft model
                    val_rmse = self._soft_rmse(X_val, y_val, slices, w, mu, m)
                    score = val_rmse
                else:
                    score = loss
                if run_best is None or score < run_best["score"]:
                    run_best = {"z": z.copy(), "score": float(score),
                                "mu": mu.copy(), "m": m.copy()}

            if best is None or run_best["score"] < best["score"]:
                best = run_best

        self.w = self._weight(best["z"])
        self.mu = best["mu"]
        self.m = best["m"]
        self._active = self.w > self.ACTIVE_W
        return self

    @property
    def active_features(self) -> list[int]:
        """Indices (into FEATURE_GROUPS) of the active (non-zero-weight) features."""
        if self._active is None:
            raise RuntimeError("fit() must be called before active_features")
        return [int(i) for i in np.flatnonzero(self._active)]

    def _soft_rmse(
        self, X, y, slices, w, mu, m
    ) -> float:
        s = group_sq_dist(X, slices, mu)
        a = softmax_rows(-np.einsum("icg,g->ic", s, w) / max(self.tau, EPS))
        yhat = a @ m
        return float(math.sqrt(np.mean((y - yhat) ** 2)))

    def build_lookup(
        self,
        df: pd.DataFrame,
        X: np.ndarray,
        y: np.ndarray,
        slices: list[tuple[int, int]],
        scaler: dict,
    ) -> "WeightedClusterModel":
        """Build the delivered hard-assignment lookup on ``df``/``X``/``y``
        using the learned weights."""
        if self.w is None:
            raise RuntimeError("fit() must be called before build_lookup()")
        mu, labels = weighted_kmeans(
            X, slices, self.w, self.n_clusters, seed=self.seed, init_mu=self.mu
        )
        self.mu_hard = mu
        self.n_hard = np.array([int((labels == c).sum()) for c in range(self.n_clusters)])
        self.m_hard = np.array([
            float(y[labels == c].mean()) if (labels == c).any() else 0.0
            for c in range(self.n_clusters)
        ])

        pred = self.predict_ratio(X)
        rel = (pred - df["ratio"].to_numpy()) / df["ratio"].to_numpy()
        self.lookup_rmse = float(100.0 * math.sqrt(np.mean(rel ** 2)))
        return self

    def predict_ratio(self, X: np.ndarray) -> np.ndarray:
        """Hard-assignment prediction of ``R2/R3``, clamped to (0, RATIO_MAX)."""
        if self.mu_hard is None or self.w is None:
            raise RuntimeError("build_lookup() must be called before predict_ratio()")
        labels = self.assign(X)
        pred = np.exp(self.m_hard[labels])
        return np.clip(pred, 1e-6, RATIO_MAX)

    def assign(self, X: np.ndarray) -> np.ndarray:
        """Hard cluster id per row."""
        if self._slices is None:
            raise RuntimeError("fit() must be called before assign()")
        return hard_assign(X, self._slices, self.w, self.mu_hard)


def save_lookup(
    model: WeightedClusterModel,
    scaler: dict,
    slices: list[tuple[int, int]],
    npz_path,
) -> None:
    """Persist the machine-readable model so new rows can be predicted."""
    s0 = np.array([a for a, _ in slices], dtype=int)
    s1 = np.array([b for _, b in slices], dtype=int)
    mean = np.array([scaler[n][0] for n in GROUP_NAMES if n in scaler], dtype=float)
    std = np.array([scaler[n][1] for n in GROUP_NAMES if n in scaler], dtype=float)
    np.savez(
        npz_path,
        w=model.w,
        mu=model.mu_hard,
        m=model.m_hard,
        n=model.n_hard,
        n_clusters=model.n_clusters,
        tau=model.tau,
        beta=model.beta,
        active=np.array(model.active_features, dtype=int) if model._active is not None
        else np.array([], dtype=int),
        lookup_rmse=model.lookup_rmse,
        scaler_mean=mean,
        scaler_std=std,
        scaler_groups=np.array([n for n in GROUP_NAMES if n in scaler]),
        slices_start=s0,
        slices_end=s1,
    )


def load_lookup(npz_path) -> dict:
    data = np.load(npz_path, allow_pickle=True)
    scaler = {
        str(g): (float(data["scaler_mean"][i]), float(data["scaler_std"][i]))
        for i, g in enumerate(data["scaler_groups"])
    }
    slices = list(zip(data["slices_start"].tolist(), data["slices_end"].tolist()))
    active = data["active"].tolist() if "active" in data.files else []
    return {
        "w": data["w"],
        "mu": data["mu"],
        "m": data["m"],
        "n": data["n"],
        "n_clusters": int(data["n_clusters"]),
        "scaler": scaler,
        "slices": slices,
        "active": [int(i) for i in active],
        "lookup_rmse": float(data["lookup_rmse"]) if "lookup_rmse" in data.files else None,
    }


def predict_lookup(params: dict, df: pd.DataFrame) -> pd.Series:
    """Apply a saved lookup to a prepared DataFrame (same schema as ``prepare``)."""
    X, _ = encode(df, params["scaler"], group_indices=params["active"] or None)
    labels = hard_assign(X, params["slices"], params["w"], params["mu"])
    pred = np.exp(params["m"][labels])
    return pd.Series(np.clip(pred, 1e-6, RATIO_MAX), index=df.index)

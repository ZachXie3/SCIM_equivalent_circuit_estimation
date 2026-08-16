"""End-to-end cluster + NN R2/R3 analysis (cluster-plan.md section 6).

Pipeline:
    load data/eq_parameters.csv
    -> prepare (drop slip<=0, R2/R3>=1; add slip/ratio/ordinals)
    -> stratified train/val/test split (by R2/R3 quartiles)
    -> encode features (scaler fit on train)
    -> beta sweep on validation (feature-count penalty strength)
    -> delivered model: refit on the active features only, build the
       hard-assignment lookup on train
    -> score test: cluster model vs baselines (unweighted cluster,
       slip+HP power law, per-pole Model D, slip-segment table)
       -- every method fitted on the train split, scored on train/val/test
    -> feature ablation (each feature removed, refit)
    -> write reports/cluster_report.md, plots, cluster_lookup.csv, model.npz

Optional:
    --sweep   also loop K = 4..8 and plot test RMSE vs K (plan section 5,
              later work - off by default)
    --k       number of clusters (default 6)

Usage:
    python -m typical_value.cluster.cluster_analysis
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from typical_value.cluster import cluster_model as cm  # noqa: E402
from typical_value.r2r3 import constant_bin as cb  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA_CSV = ROOT / "data" / "eq_parameters.csv"
REPORT_MD = ROOT / "reports" / "cluster_report.md"
PLOTS_DIR = ROOT / "reports" / "plots"
LOOKUP_CSV = ROOT / "reports" / "cluster_lookup.csv"
MODEL_NPZ = ROOT / "reports" / "cluster_model.npz"

TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.70, 0.15, 0.15


# ---------------------------------------------------------------------------
# Splits + metrics
# ---------------------------------------------------------------------------


def stratified_split(df: pd.DataFrame, seed: int = 42):
    """Train/val/test split stratified by R2/R3 quartiles."""
    q = pd.qcut(df["ratio"], 4, labels=False, duplicates="drop")
    rng = np.random.default_rng(seed)
    tr, va, te = [], [], []
    for level in q.sort_values().unique():
        idx = np.flatnonzero((q == level).to_numpy())
        rng.shuffle(idx)
        n = len(idx)
        n_tr = int(round(n * TRAIN_FRAC))
        n_va = int(round(n * VAL_FRAC))
        tr.append(idx[:n_tr])
        va.append(idx[n_tr:n_tr + n_va])
        te.append(idx[n_tr + n_va:])
    return (df.iloc[np.concatenate(tr)].reset_index(drop=True).copy(),
            df.iloc[np.concatenate(va)].reset_index(drop=True).copy(),
            df.iloc[np.concatenate(te)].reset_index(drop=True).copy())


def rel_metrics(true_ratio: np.ndarray, pred: np.ndarray) -> dict:
    """Relative-error metrics on R2/R3 (percent)."""
    rel = (pred - true_ratio) / true_ratio
    return {
        "RMSE": float(100.0 * np.sqrt(np.mean(rel ** 2))),
        "MAE": float(100.0 * np.mean(np.abs(rel))),
        "P10": float(100.0 * np.percentile(rel, 10)),
        "P50": float(100.0 * np.percentile(rel, 50)),
        "P90": float(100.0 * np.percentile(rel, 90)),
    }


# ---------------------------------------------------------------------------
# Baselines (fit on train, scored on train/val/test)
# ---------------------------------------------------------------------------


def fit_power_bhp(df: pd.DataFrame) -> np.ndarray:
    X = np.column_stack([np.ones(len(df)), np.log(df["slip"]), np.log(df["HorsePower"])])
    b, *_ = np.linalg.lstsq(X, np.log(df["ratio"]), rcond=None)
    return b


def predict_power_bhp(b: np.ndarray, df: pd.DataFrame) -> np.ndarray:
    X = np.column_stack([np.ones(len(df)), np.log(df["slip"]), np.log(df["HorsePower"])])
    return np.exp(X @ b)


def fit_power_d(df: pd.DataFrame):
    params = {}
    for pole, g in df.groupby("PoleSpeed"):
        X = np.column_stack([np.ones(len(g)), np.log(g["slip"]), np.log(g["HorsePower"])])
        coef, *_ = np.linalg.lstsq(X, np.log(g["ratio"]), rcond=None)
        params[int(pole)] = coef
    return params, fit_power_bhp(df)


def predict_power_d(params, fb: np.ndarray, df: pd.DataFrame) -> np.ndarray:
    pred = np.empty(len(df))
    for pole, g in df.groupby("PoleSpeed"):
        c = params.get(int(pole), fb)
        X = np.column_stack([np.ones(len(g)), np.log(g["slip"]), np.log(g["HorsePower"])])
        pred[g.index] = np.exp(X @ c)
    return pred


def unweighted_cluster_model(df_fit, Xfit, yfit, slices, n_clusters, seed):
    """Plain K-means (all feature weights equal, w = 1) as a baseline."""
    m = cm.WeightedClusterModel(n_clusters=n_clusters, seed=seed, n_restarts=1)
    m.w = np.ones(len(slices))
    m._slices = slices
    mu, _ = cm.kmeans2(Xfit, n_clusters, minit="++", iter=10, seed=seed)
    m.mu = mu
    m.build_lookup(df_fit, Xfit, yfit, slices, {})
    return m


def run_ablation(df_tr, df_va, df_te, scaler, seed, k, slices, Xtr, Xva, Xte, ytr, yva):
    """Refit with each feature group removed (no count penalty, so it isolates
    the pure RMSE impact); return {feature: test RMSE}."""
    out = {}
    for gi in range(len(cm.FEATURE_GROUPS)):
        keep = [j for j in range(len(cm.FEATURE_GROUPS)) if j != gi]
        Xtr_i, sl_i = cm.encode(df_tr, scaler, keep)
        Xva_i, _ = cm.encode(df_va, scaler, keep)
        Xte_i, _ = cm.encode(df_te, scaler, keep)
        m = cm.WeightedClusterModel(n_clusters=k, beta=0.0, seed=seed) \
            .fit(Xtr_i, ytr, sl_i, Xva_i, yva)
        m.build_lookup(df_tr, Xtr_i, ytr, sl_i, scaler)
        pred = m.predict_ratio(Xte_i)
        out[cm.GROUP_NAMES[gi]] = rel_metrics(df_te["ratio"].to_numpy(), pred)["RMSE"]
    return out


# ---------------------------------------------------------------------------
# Beta selection (feature-count penalty strength)
# ---------------------------------------------------------------------------

BETA_GRID = [0.0, 0.05, 0.1, 0.2, 0.4, 0.8]


def select_beta(df_tr, df_va, scaler, k, seed, tol_pp=1.5):
    """Choose the penalty strength ``beta`` on the validation split.

    For each ``beta`` fit the soft model on train, prune to the active
    features (weight > ACTIVE_W), refit on the reduced encoding and build the
    hard lookup; then score the delivered model on validation. Pick the
    smallest active-feature set whose validation RMSE is within ``tol_pp`` pp
    of the best (unpenalised) model - i.e. as few features as possible without
    a major performance compromise.

    Returns ``(beta, results)`` where ``results[beta]`` holds ``active``,
    ``n_active``, ``val_rmse`` and the fitted ``model``.
    """
    ytr, yva = df_tr["y"].to_numpy(), df_va["y"].to_numpy()
    Xfull, slices = cm.encode(df_tr, scaler)
    Xva_full, _ = cm.encode(df_va, scaler)

    results = {}
    for beta in BETA_GRID:
        m = cm.WeightedClusterModel(n_clusters=k, beta=beta, seed=seed) \
            .fit(Xfull, ytr, slices, Xva_full, yva)
        active = m.active_features
        if len(active) == 0:
            # everything pruned -> degenerate, exclude from selection
            results[beta] = {"active": [], "n_active": 0,
                             "val_rmse": float("inf"), "model": None}
            continue
        Xa, slices_a = cm.encode(df_tr, scaler, active)
        Xva_a, _ = cm.encode(df_va, scaler, active)
        m2 = cm.WeightedClusterModel(n_clusters=k, beta=beta, seed=seed) \
            .fit(Xa, ytr, slices_a, Xva_a, yva)
        m2.build_lookup(df_tr, Xa, ytr, slices_a, scaler)
        val_rmse = rel_metrics(df_va["ratio"].to_numpy(), m2.predict_ratio(Xva_a))["RMSE"]
        results[beta] = {
            "active": active, "n_active": len(active),
            "val_rmse": val_rmse, "model": m2,
        }

    feasible = [b for b, r in results.items() if r["n_active"] > 0]
    best_val = min(results[b]["val_rmse"] for b in feasible)
    candidates = [b for b in feasible if results[b]["val_rmse"] <= best_val + tol_pp]
    chosen = min(candidates, key=lambda b: (results[b]["n_active"], results[b]["val_rmse"]))
    return chosen, results


# ---------------------------------------------------------------------------
# Cluster table / lookup
# ---------------------------------------------------------------------------


def cluster_rows(model, df_fit, Xfit, yfit):
    """Per-cluster rows for the report and the human-readable lookup."""
    labels = model.assign(Xfit)
    rows = []
    for c in range(model.n_clusters):
        sel = df_fit.iloc[np.flatnonzero(labels == c)]
        if len(sel) == 0:
            continue
        rel = (np.exp(model.m_hard[c]) - sel["ratio"]) / sel["ratio"]
        rows.append({
            "cluster": c,
            "n": int(len(sel)),
            "ratio": float(np.exp(model.m_hard[c])),
            "rmse_rel": float(100.0 * np.sqrt(np.mean(rel ** 2))),
            "slip_mean": float(sel["slip"].mean()),
            "hp_mean": float(sel["HorsePower"].mean()),
            "pole_mode": str(sel["PoleSpeed"].mode().iloc[0]),
            "nema_design": str(sel["NemaDesign"].mode().iloc[0])
                           if sel["NemaDesign"].notna().any() else "missing",
            "nema_kva": str(sel["NemaKVACode"].mode().iloc[0])
                        if sel["NemaKVACode"].notna().any() else "missing",
            "bar_material": str(sel["BarMaterial"].mode().iloc[0]),
            "name": str(sel["Name"].mode().iloc[0]),
        })
    return rows


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_weights(weights: dict, path: Path) -> None:
    names = list(weights)
    vals = [weights[n] for n in names]
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#4C72B0" if v > 0 else "#C44E52" for v in vals]
    ax.bar(names, vals, color=colors)
    ax.set_ylabel("learned weight w")
    ax.axhline(cm.WeightedClusterModel.ACTIVE_W, color="k", ls="--", lw=0.8,
               alpha=0.5, label="active threshold")
    ax.legend()
    ax.set_title("Learned per-feature weights (red = pruned)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_parity(true_ratio, pred, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(true_ratio, pred, s=8, alpha=0.35, color="#4C72B0")
    lo = min(true_ratio.min(), pred.min())
    hi = max(true_ratio.max(), pred.max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1, label="y = x")
    ax.set_xlabel("true R2/R3")
    ax.set_ylabel("predicted R2/R3")
    ax.set_title("Cluster model - test parity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_k_sweep(sweep_results: dict, path: Path) -> None:
    ks = sorted(sweep_results)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ks, [sweep_results[k]["train"] for k in ks], "o-", label="train RMSE")
    ax.plot(ks, [sweep_results[k]["test"] for k in ks], "o-", label="test RMSE")
    ax.set_xlabel("clusters K")
    ax.set_ylabel("RMSE (%)")
    ax.set_title("Cluster count sweep (train vs test)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def write_report(args, prep, model, weights, ablation, cluster_tbl, metrics_all,
                 sweep_results, beta_sel, beta_results) -> None:
    lines = []
    w = lines.append
    w("# Cluster + NN R2/R3 Report (cluster-plan.md)")
    w("")
    w(f"> Auto-generated by `typical_value/cluster/cluster_analysis.py` "
      f"(K = {args.k}, seed = {args.seed}, tau = {model.tau}).")
    w("")
    w("## 1. Data preparation")
    w("")
    w(f"- Source: `{DATA_CSV.name}` ({prep['total']} rows).")
    w(f"- Dropped: `slip <= 0` - {prep['n_slip']}; `R2/R3 >= 1.0` - {prep['n_ratio1']}; "
      f"kept **{prep['kept']}**.")
    w(f"- Split (stratified by R2/R3 quartiles): "
      f"train {prep['n_train']} / validation {prep['n_val']} / test {prep['n_test']}.")
    w("")

    active = set(beta_results[beta_sel]["active"])
    w("## 2. Learned feature weights (NN output)")
    w("")
    w("| Feature | weight | active? | ablation test RMSE (%) | vs full (pp) |")
    w("|---|---|---:|---:|---:|")
    full_rmse = metrics_all["cluster"]["test"]["RMSE"]
    for i, n in enumerate(cm.GROUP_NAMES):
        abl = ablation.get(n, float("nan"))
        mark = "yes" if i in active else "no (pruned)"
        w(f"| {n} | {weights[n]:.3f} | {mark} | {abl:.1f} | {abl - full_rmse:+.1f} |")
    w("")
    w(f"- Full (pruned) model test RMSE: **{full_rmse:.1f}%** with "
      f"**{len(active)}/{len(cm.GROUP_NAMES)} active features** "
      f"({', '.join(cm.GROUP_NAMES[i] for i in sorted(active))}).")
    w("")

    w("### 2.1 Beta sweep (feature-count penalty)")
    w("")
    w("`beta` = cost of one active feature, as a fraction of the reference "
      "RMSE. Chosen on validation: smallest active-feature set whose "
      f"validation RMSE stays within {args.tol_pp:.1f} pp of the best.")
    w("")
    w("| beta | active features | validation RMSE (%) |")
    w("|---|---:|---:|")
    for b in BETA_GRID:
        r = beta_results[b]
        mark = "  <- selected" if b == beta_sel else ""
        val = f"{r['val_rmse']:.1f}" if r["n_active"] else "inf"
        w(f"| {b} | {r['n_active']} | {val} |{mark}")
    w("")

    w("## 3. Cluster lookup")
    w("")
    w("| Cluster | n | R2/R3 | within RMSE (%) | slip | HP | Poles | NemaDesign | KVA | Material | Name |")
    w("|---|---:|---:|---:|---|---|---|---|---|---|---|")
    for r in cluster_tbl:
        w(f"| {r['cluster']} | {r['n']} | {r['ratio']:.3f} | {r['rmse_rel']:.1f} | "
          f"{r['slip_mean']:.3f} | {r['hp_mean']:.0f} | {r['pole_mode']} | {r['nema_design']} | "
          f"{r['nema_kva']} | {r['bar_material']} | {r['name']} |")
    w("")

    w("## 4. Performance by split (all methods fitted on the train split)")
    w("")
    w("RMSE (%) on train (in-sample), validation, and test. Every method is "
      "fitted on the **train** split only; the test split is held out.")
    w("")
    w("| Method | Train RMSE (%) | Train MAE (%) | Validation RMSE (%) | Validation MAE (%) | Test RMSE (%) | Test MAE (%) |")
    w("|---|---:|---:|---:|---:|---:|---:|")
    for name, m in metrics_all.items():
        w(f"| {name} | {m['train']['RMSE']:.1f} | {m['train']['MAE']:.1f} | "
          f"{m['validation']['RMSE']:.1f} | {m['validation']['MAE']:.1f} | "
          f"{m['test']['RMSE']:.1f} | {m['test']['MAE']:.1f} |")
    w("")
    w(f"- The cluster model uses the validation split to select its weights, so "
      f"its validation number is mildly optimistic; the test number is unbiased. "
      f"Baselines have no selection step.")
    w("")

    w("## 5. Test-set detail")
    w("")
    w("| Method | RMSE (%) | MAE (%) | P10 | P50 | P90 |")
    w("|---|---:|---:|---:|---:|---:|")
    for name, m in metrics_all.items():
        mm = m["test"]
        w(f"| {name} | {mm['RMSE']:.1f} | {mm['MAE']:.1f} | "
          f"{mm['P10']:+.1f} | {mm['P50']:+.1f} | {mm['P90']:+.1f} |")
    w("")

    if sweep_results:
        w("## 6. Cluster-count sweep (test RMSE)")
        w("")
        w("| K | train RMSE (%) | test RMSE (%) |")
        w("|---|---:|---:|")
        for k in sorted(sweep_results):
            r = sweep_results[k]
            w(f"| {k} | {r['train']:.1f} | {r['test']:.1f} |")
        w("")
        w("See `reports/plots/cluster_rmse_vs_k.png`.")
        w("")

    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {REPORT_MD}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_k_sweep(df_tr, df_va, df_te, scaler, seed, beta, Xtr, Xva, Xte, ytr, yva, yte):
    out = {}
    for k in range(4, 9):
        slices = cm.encode(df_tr, scaler)[1]
        m = cm.WeightedClusterModel(n_clusters=k, beta=beta, seed=seed) \
            .fit(Xtr, ytr, slices, Xva, yva)
        m.build_lookup(df_tr, Xtr, ytr, slices, scaler)
        out[k] = {
            "train": rel_metrics(df_tr["ratio"].to_numpy(), m.predict_ratio(Xtr))["RMSE"],
            "test": rel_metrics(df_te["ratio"].to_numpy(), m.predict_ratio(Xte))["RMSE"],
        }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=6, help="number of clusters (default 6)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tol-pp", type=float, default=1.5,
                    help="beta selection: allowed validation RMSE loss (pp) vs the "
                         "unpenalised best before a feature is pruned (default 1.5)")
    ap.add_argument("--sweep", action="store_true",
                    help="also loop K=4..8 and add the RMSE-vs-K section (plan section 5, "
                         "later work)")
    args = ap.parse_args(argv)

    raw = pd.read_csv(DATA_CSV)
    df = cm.prepare(raw)

    n_slip = int((((raw["SRPM"] - raw["RPM"]) / raw["SRPM"]) <= 0).sum())
    n_ratio1 = int((raw["R2"] / raw["R3"] >= 1.0).sum())
    prep = {
        "total": len(raw), "n_slip": n_slip, "n_ratio1": n_ratio1, "kept": len(df),
        "n_train": 0, "n_val": 0, "n_test": 0,
    }

    df_tr, df_va, df_te = stratified_split(df, args.seed)
    prep.update(n_train=len(df_tr), n_val=len(df_va), n_test=len(df_te))

    scaler = cm.fit_scaler(df_tr)
    Xtr, slices = cm.encode(df_tr, scaler)
    Xva, _ = cm.encode(df_va, scaler)
    Xte, _ = cm.encode(df_te, scaler)
    ytr, yva, yte = (df_tr["y"].to_numpy(), df_va["y"].to_numpy(), df_te["y"].to_numpy())

    # --- weighted cluster model with feature-count penalty ---
    # beta is chosen on validation (fewest active features within tol_pp of
    # the best); the delivered model is refit on the active features only.
    beta_sel, beta_results = select_beta(df_tr, df_va, scaler, args.k,
                                         args.seed, tol_pp=args.tol_pp)
    model = beta_results[beta_sel]["model"]
    active = beta_results[beta_sel]["active"]
    Xtr_a, slices_a = cm.encode(df_tr, scaler, active)
    Xva_a, _ = cm.encode(df_va, scaler, active)
    Xte_a, _ = cm.encode(df_te, scaler, active)

    # Full-length weight map: active features keep their learned weight,
    # pruned features are 0.
    w_by_group = dict(zip(active, model.w.tolist()))
    weights = {cm.GROUP_NAMES[i]: float(w_by_group.get(i, 0.0))
               for i in range(len(cm.GROUP_NAMES))}

    def score(name: str, pred_train, pred_val, pred_test) -> None:
        metrics_all[name] = {
            "train": rel_metrics(df_tr["ratio"].to_numpy(), pred_train),
            "validation": rel_metrics(df_va["ratio"].to_numpy(), pred_val),
            "test": rel_metrics(df_te["ratio"].to_numpy(), pred_test),
        }

    # Every method below is fitted on the TRAIN split only, then scored on
    # train / validation / test so the comparison is apples-to-apples.

    # --- weighted cluster model (delivered, pruned to active features) ---
    metrics_all = {}
    score("cluster",
          model.predict_ratio(Xtr_a), model.predict_ratio(Xva_a),
          model.predict_ratio(Xte_a))

    # --- unweighted cluster baseline (all weights equal) ---
    unw = unweighted_cluster_model(df_tr, Xtr, ytr, slices, args.k, args.seed)
    score("cluster unweighted",
          unw.predict_ratio(Xtr), unw.predict_ratio(Xva), unw.predict_ratio(Xte))

    # --- slip + HP power law ---
    bhp = fit_power_bhp(df_tr)
    score("slip+HP power law",
          predict_power_bhp(bhp, df_tr), predict_power_bhp(bhp, df_va),
          predict_power_bhp(bhp, df_te))

    # --- per-pole Model D ---
    pd_params, pd_fb = fit_power_d(df_tr)
    score("per-pole Model D",
          predict_power_d(pd_params, pd_fb, df_tr), predict_power_d(pd_params, pd_fb, df_va),
          predict_power_d(pd_params, pd_fb, df_te))

    # --- slip-segment table ---
    seg = cb.fit_slip_segments(df_tr.assign(s_fl=df_tr["slip"]))
    score("slip-segment table",
          cb.predict_slip_segments(seg, df_tr["slip"]).to_numpy(),
          cb.predict_slip_segments(seg, df_va["slip"]).to_numpy(),
          cb.predict_slip_segments(seg, df_te["slip"]).to_numpy())

    # --- ablation ---
    ablation = run_ablation(df_tr, df_va, df_te, scaler, args.seed, args.k, slices,
                            Xtr, Xva, Xte, ytr, yva)

    # --- artifacts ---
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_weights(weights, PLOTS_DIR / "cluster_weights.png")
    plot_parity(df_te["ratio"].to_numpy(), model.predict_ratio(Xte_a),
                PLOTS_DIR / "cluster_parity.png")

    cluster_tbl = cluster_rows(model, df_tr, Xtr_a, ytr)
    pd.DataFrame(cluster_tbl).to_csv(LOOKUP_CSV, index=False)

    cm.save_lookup(model, scaler, slices_a, MODEL_NPZ)

    sweep_results = run_k_sweep(df_tr, df_va, df_te, scaler, args.seed, beta_sel,
                                Xtr, Xva, Xte, ytr, yva, yte) if args.sweep else {}
    if sweep_results:
        plot_k_sweep(sweep_results, PLOTS_DIR / "cluster_rmse_vs_k.png")

    write_report(args, prep, model, weights, ablation, cluster_tbl, metrics_all,
                 sweep_results, beta_sel, beta_results)
    print(f"Wrote {LOOKUP_CSV}")
    print(f"Wrote {MODEL_NPZ}")
    print(f"Wrote plots -> {PLOTS_DIR}")
    print(f"beta = {beta_sel} -> active features ({len(active)}/{len(cm.GROUP_NAMES)}): "
          f"{', '.join(cm.GROUP_NAMES[i] for i in active)}")
    header = f"{'Method':<22}{'train':>10}{'validation':>12}{'test':>10}"
    print(header)
    print("-" * len(header))
    for name, mm in metrics_all.items():
        print(f"{name:<22}{mm['train']['RMSE']:>9.1f}%{mm['validation']['RMSE']:>11.1f}%"
              f"{mm['test']['RMSE']:>9.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

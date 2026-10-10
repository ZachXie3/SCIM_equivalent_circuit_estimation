"""Stage 5 validation harness (plan.md §5).

Runs the Stages 1-4 equivalent-circuit estimation pipeline over
``data/eq_parameters.csv`` and compares the predicted per-phase parameters
against the dataset's ``R2, R3, X1, X2, X3, Xm`` columns.

Design (plan.md §5.4 - matrix-driven, no per-row Python loop):

* Stage 1 + Stage 2  : closed-form vectorised over the whole table.
* Stage 3 (X1, X2)   : vectorised Gauss-Newton least squares over all rows
                       at once, with a masked backtracking line search.
* Stage 4 outer loop : pure array updates (Xm = X0 - X1, X3 = XLR - X1).
* Reference path     : per-row ``EquivalentCircuitEstimator.fit()`` on a
                       subsample, cross-checked against the vectorised path
                       and used for the runtime comparison.

Inputs (plan §5.2) are the real measurement columns — ``PowerFactor100``
(PF_FL), ``LockedRotorAmps`` (I_LR), ``NoLoadAmps`` (I_0), ``NoLoadWatts``
(P_0), ``LockedRotorTorque`` (T_LR as % of full-load torque, /100), 
``WindingResistAt105`` (R1_hot) — plus the nameplate columns. Only rows with
``R2/R3 < 1`` and ``slip > 0`` are used (§5.1).

Per-phase convention: the circuit values are the wye-equivalent per-phase
circuit (delta-to-wye ``Z_y = Z_d/3``, applied by ``Jconn`` in
``eq_calculator.py``), so ``V_ph = V_LL/sqrt(3)`` and ``I_ph = I_line`` for
BOTH connections.

Two run modes (§5.2.2):

* ``prior``        — R2 from the empirical R2/R3 prior (§2.4): production view,
                     prior error included.
* ``ground-truth`` — R2 = dataset R2 (R2_override): the §2.5 experiment that
                     isolates the circuit-solve error.

Usage:
    python validate.py                    # full run, both modes
    python validate.py --mode prior       # only the prior mode
    python validate.py --subsample 200    # also cross-check reference on N rows
    python validate.py --report-only      # regenerate report from saved CSVs
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import equivalent_circuit as ec  # noqa: E402

ROOT = Path(__file__).resolve().parent
RESULTS_CSV = ROOT / "data" / "eq_parameters.csv"
REPORT_MD = ROOT / "reports" / "validation_report.md"

T_AMB = 25.0
T_RISE = 80.0
T_HOT = T_AMB + T_RISE
K1 = ec.K1_COPPER
HP_TO_WATT = ec.HP_TO_WATT

REFERENCE_TOL_REL = 0.02   # vectorised vs per-row reference (plan §5.4)
COVERAGE_MIN = 0.90        # share of rows required to converge
OUTLIER_THRESHOLD = 0.15   # |rel err| above this -> outlier (plan §5.3)

MODES = ("prior", "ground-truth")
METRIC_PARAMS = ("R2", "R3", "X1", "X2", "X3", "Xm")

VOLTAGE_EDGES = [0.0, 500.0, 1000.0, 2000.0, 5000.0, np.inf]
VOLTAGE_LABELS = ["<500", "500-999", "1000-1999", "2000-4999", "5000+"]
HP_EDGES = [0.0, 100.0, 500.0, 1000.0, 2000.0, np.inf]
HP_LABELS = ["<100", "100-499", "500-999", "1000-1999", "2000+"]


# ---------------------------------------------------------------------------
# Row filter (plan §5.1)
# ---------------------------------------------------------------------------

def valid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Rows usable for validation: ``slip > 0`` and ``R2/R3 < 1`` (a ratio >= 1
    violates the physics bound ``R2 <= R3`` and is not valid data)."""
    slip = (df["SRPM"] - df["RPM"]) / df["SRPM"]
    ratio = df["R2"] / df["R3"]
    df = df[(slip > 0) & (ratio < 1.0)]
    return df.reset_index(drop=True).copy()


# ---------------------------------------------------------------------------
# Vectorised Stage 1 + Stage 2 (plan §5.4)
# ---------------------------------------------------------------------------

def _is_delta(df: pd.DataFrame) -> np.ndarray:
    conn = df["Connection"].astype(str).str.strip().str.upper()
    return conn.str.startswith("D").to_numpy(dtype=bool)


def stage1_stage2(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Closed-form per-phase quantities over the whole table (vectorised).

    Uses the real measurement columns (plan §5.2): ``PowerFactor100``,
    ``LockedRotorAmps``, ``NoLoadAmps``, ``NoLoadWatts``,
    ``WindingResistAt105``, ``LockedRotorTorque``.

    Per-phase convention (wye-equivalent): ``V_ph = V_LL/sqrt(3)`` and
    ``I_ph = I_line`` for BOTH connections — the circuit values are the
    wye-equivalent per-phase circuit (delta-to-wye: ``Z_y = Z_d/3``, applied
    by ``Jconn`` in ``eq_calculator.py``). ``T_LR`` (pu of full-load torque) is
    ``LockedRotorTorque/100`` — the column is a percentage of full-load torque.
    """
    is_delta = _is_delta(df)
    sq3 = np.sqrt(3.0)

    v_ll = df["Voltage"].to_numpy(dtype=float)
    v_ph = v_ll / sq3
    i_fl_ph = df["Amps"].to_numpy(dtype=float)
    i_0_ph = df["NoLoadAmps"].to_numpy(dtype=float)
    i_lr_ph = df["LockedRotorAmps"].to_numpy(dtype=float)

    n_s = 120.0 * df["Frequency"].to_numpy(dtype=float) / df["PoleSpeed"].to_numpy(dtype=float)
    omega_s = 2.0 * np.pi * n_s / 60.0
    s_fl = (n_s - df["RPM"].to_numpy(dtype=float)) / n_s
    n_rpm = df["RPM"].to_numpy(dtype=float)
    t_fl = df["HorsePower"].to_numpy(dtype=float) * HP_TO_WATT / np.maximum(
        2.0 * np.pi * n_rpm / 60.0, ec.EPS)

    jconn = np.where(is_delta, 3.0, 1.0)
    r1_105 = df["WindingResistAt105"].to_numpy(dtype=float) / jconn
    r1_cold_feed = r1_105 * (T_AMB + K1) / (T_HOT + K1)
    r1_hot = r1_105

    # Real measurements: PF fraction, locked-rotor line current, no-load line
    # current, no-load power, and locked-rotor torque as % of full-load torque.
    pf_fl = df["PowerFactor100"].to_numpy(dtype=float)
    i_lr_line = i_lr_ph
    i_0_line = i_0_ph
    p_0 = df["NoLoadWatts"].to_numpy(dtype=float)
    t_lr_pu = df["LockedRotorTorque"].to_numpy(dtype=float) / 100.0

    return {
        "v_ph": v_ph, "i_fl_ph": i_fl_ph, "i_0_ph": i_0_ph, "i_lr_ph": i_lr_ph,
        "i_fl_line": i_fl_ph,
        "i_lr_line": i_lr_line, "i_0_line": i_0_line, "pf_fl": pf_fl, "p_0": p_0,
        "t_lr_pu": t_lr_pu,
        "n_s": n_s, "omega_s": omega_s, "s_fl": s_fl, "t_fl": t_fl,
        "r1_cold_feed": r1_cold_feed, "r1_hot": r1_hot, "is_delta": is_delta,
    }


def stage1_backsolve(d: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Stage-1 back-solves R3, XLR, X0 from the real test data."""
    i_lr_ph = d["i_lr_ph"]
    i_0_ph = d["i_0_ph"]

    r3_est = (d["t_lr_pu"] * d["t_fl"] * d["omega_s"]
              / np.maximum(3.0 * i_lr_ph**2, ec.EPS))

    r_lr = d["r1_cold_feed"] + r3_est
    z_lr = d["v_ph"] / np.maximum(i_lr_ph, ec.EPS)
    z_sq = z_lr**2 - r_lr**2
    xlr = np.where(z_sq > 0.0, np.sqrt(np.maximum(z_sq, 0.0)), ec.EPS)

    s0 = 3.0 * d["v_ph"] * i_0_ph
    q0 = np.sqrt(np.maximum(s0**2 - d["p_0"]**2, 0.0))
    x0 = 3.0 * d["v_ph"]**2 / np.maximum(q0, ec.EPS)

    return {"r3_est": r3_est, "xlr": xlr, "x0": x0}


def _build_dyn(df: pd.DataFrame) -> pd.DataFrame:
    """One flat table of per-row serving data for the whole pipeline."""
    st = stage1_stage2(df)
    dyn = {**st}
    for k, v in (stage1_backsolve(dyn)).items():
        dyn[k] = v
    dyn["pole"] = df["PoleSpeed"].to_numpy(dtype=int)
    dyn["hp"] = df["HorsePower"].to_numpy(dtype=float)
    dyn["f"] = df["Frequency"].to_numpy(dtype=float)
    dyn["n_FL"] = df["RPM"].to_numpy(dtype=float)
    dyn["design_id"] = df.index.to_numpy()
    dyn["r2_gt"] = df["R2"].to_numpy(dtype=float)   # per-phase ground truth
    dyn["Connection"] = df["Connection"].astype(str).to_numpy()
    return pd.DataFrame(dyn)


def r2_prior_series(dyn: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Set the R2 used by the solver, by mode (plan §5.2.2).

    ``prior``        — R2 = R3_est * R2R3_MODEL_D prior (production view).
    ``ground-truth`` — R2 = dataset R2 (the §2.5 experiment).
    """
    if mode == "ground-truth":
        r3_est = dyn["r3_est"].to_numpy(dtype=float)
        r2 = np.where(r3_est > 0.0,
                      np.minimum(dyn["r2_gt"].to_numpy(dtype=float), r3_est), 0.0)
        dyn["r2_prior"] = r2
        dyn["ratio"] = np.nan
        return dyn

    poles = dyn["pole"].to_numpy(dtype=int)
    hp = dyn["hp"].to_numpy(dtype=float)
    s = dyn["s_fl"].to_numpy()

    a = np.zeros(len(poles)); b = np.zeros(len(poles)); c = np.zeros(len(poles))
    for pole, coef in ec.R2R3_MODEL_D.items():
        idx = poles == pole
        a[idx] = coef["a"]; c[idx] = coef["c"]; b[idx] = coef["b"]
    seen = np.isin(poles, list(ec.R2R3_MODEL_D))
    a = np.where(seen, a, ec.R2R3_MODEL_FALLBACK["a"])
    b = np.where(seen, b, ec.R2R3_MODEL_FALLBACK["b"])
    c = np.where(seen, c, 0.0)

    log_ok = (s > 0) & (hp > 0) & (s <= 1.0)
    ratio = np.zeros(len(poles))
    ratio[log_ok] = np.exp(b[log_ok] + a[log_ok] * np.log(s[log_ok])
                           + c[log_ok] * np.log(hp[log_ok]))
    r3_est = dyn["r3_est"].to_numpy(dtype=float)
    r2 = np.where(r3_est > 0.0, np.minimum(r3_est * ratio, r3_est), 0.0)
    dyn["r2_prior"] = r2
    dyn["ratio"] = ratio
    return dyn


# ---------------------------------------------------------------------------
# Vectorised Stage 3 + Stage 4 (plan §5.4)
# ---------------------------------------------------------------------------

def _resid(x1: np.ndarray, x2: np.ndarray, dyn: pd.DataFrame,
           xm: np.ndarray):
    """Full-load residuals (n, 3): current Re, current Im, torque."""
    s = np.maximum(dyn["s_fl"].to_numpy(), 1e-9)
    v = dyn["v_ph"].to_numpy()
    r1 = dyn["r1_hot"].to_numpy()
    iph = dyn["i_fl_ph"].to_numpy()
    tf = dyn["t_fl"].to_numpy()
    om = dyn["omega_s"].to_numpy()
    r2 = dyn["r2_prior"].to_numpy()
    pf = dyn["pf_fl"].to_numpy()
    phi = np.arccos(np.clip(pf, -1.0, 1.0))
    ivec = iph * (np.cos(phi) - 1j * np.sin(phi))

    zm = 1j * xm
    z2 = r2 / s + 1j * x2
    zpar = zm * z2 / (zm + z2)
    zfl = r1 + 1j * x1 + zpar
    ipr = v / zfl
    re = (ipr.real - ivec.real) / np.maximum(iph, 1e-12)
    im = (ipr.imag - ivec.imag) / np.maximum(iph, 1e-12)

    i2 = ivec * zm / np.maximum(np.abs(zm + z2), 1e-12)
    tc = 3.0 * np.abs(i2)**2 * (1.0 - s) / np.maximum(s, 1e-12) * r2 / np.maximum(om, 1e-12)
    t = (tc - tf) / np.maximum(tf, 1e-12)

    bad = (s <= 1e-6) | (xm <= 0.0) | (x2 <= 0.0) | (x1 <= 0.0)
    re = np.where(bad, 1e12, re); im = np.where(bad, 1e12, im); t = np.where(bad, 1e12, t)
    return np.stack([re, im, t], axis=1)


def _gn_run(x1, x2, dyn, xm, x1hi, x2hi, max_iter):
    """One vectorised Gauss-Newton run from a given start; returns best & cost."""
    f = _resid(x1, x2, dyn, xm)
    cost = np.einsum("ij,ij->i", f, f)
    x1b, x2b, cb = x1, x2, cost
    for _ in range(max_iter):
        h1 = 1e-4 * np.maximum(np.abs(x1), 1e-6)
        f1 = _resid(x1 + h1, x2, dyn, xm)
        j1 = (f1 - f) / h1[:, None]

        h2 = 1e-4 * np.maximum(np.abs(x2), 1e-6)
        f2 = _resid(x1, x2 + h2, dyn, xm)
        j2 = (f2 - f) / h2[:, None]

        g11 = j1[:, 0]**2 + j1[:, 1]**2 + j1[:, 2]**2
        g22 = j2[:, 0]**2 + j2[:, 1]**2 + j2[:, 2]**2
        g12 = j1[:, 0]*j2[:, 0] + j1[:, 1]*j2[:, 1] + j1[:, 2]*j2[:, 2]
        d1 = j1[:, 0]*f[:, 0] + j1[:, 1]*f[:, 1] + j1[:, 2]*f[:, 2]
        d2 = j2[:, 0]*f[:, 0] + j2[:, 1]*f[:, 1] + j2[:, 2]*f[:, 2]

        det = g11 * g22 - g12 * g12
        reg = 1e-9 * (g11 + g22)
        dets = np.where(np.abs(det) < 1e-30, reg, det)
        x1s = (d1 * g22 - d2 * g12) / np.maximum(dets, 1e-30)
        x2s = (d2 * g11 - d1 * g12) / np.maximum(dets, 1e-30)

        improved = np.zeros(len(dyn), dtype=bool)
        for scale in (1.0, 0.5, 0.25, 0.125, 1e-6):
            n1 = np.clip(x1 - scale * x1s, 1e-9, x1hi)
            n2 = np.clip(x2 - scale * x2s, 1e-9, x2hi)
            nf = _resid(n1, n2, dyn, xm)
            ncost = np.einsum("ij,ij->i", nf, nf)
            better = (ncost < cost) & ~improved
            x1 = np.where(better, n1, x1)
            x2 = np.where(better, n2, x2)
            improved |= better
        if not improved.any():
            break
        f = _resid(x1, x2, dyn, xm)
        cost = np.einsum("ij,ij->i", f, f)
    better = cost < cb
    x1b = np.where(better, x1, x1b)
    x2b = np.where(better, x2, x2b)
    cb = np.where(better, cost, cb)
    return x1b, x2b, cb


def solve_x1x2_vector(dyn: pd.DataFrame, xm: np.ndarray, max_iter: int = 30) -> dict:
    """Vectorised Gauss-Newton for (X1, X2), all rows at once.

    Mirrors the per-row reference solver (plan §5.4): the same {0.02, 0.05,
    0.15} starting-fraction grid over the same bounds, keeping the best
    (lowest residual cost) per row. Runs the whole grid at once via numpy.
    """
    n = len(dyn)
    x0 = dyn["x0"].to_numpy()
    xlr = dyn["xlr"].to_numpy()
    x1hi = 0.999 * np.minimum(np.maximum(xm, 1e-9), np.maximum(xlr, 1e-9))
    x2hi = np.maximum(2.0 * np.maximum(x0, 0.0), 1.0)

    x1 = np.zeros(n); x2 = np.zeros(n); cb = np.full(n, np.inf)
    for x1f in (0.02, 0.05, 0.15):
        for x2f in (0.02, 0.05, 0.15):
            s1 = np.clip(x1f * x1hi, 1e-9, x1hi)
            s2 = np.clip(x2f * x2hi, 1e-9, x2hi)
            b1, b2, bc = _gn_run(s1, s2, dyn, xm, x1hi, x2hi, max_iter)
            better = bc < cb
            x1 = np.where(better, b1, x1)
            x2 = np.where(better, b2, x2)
            cb = np.where(better, bc, cb)
    return {"X1": x1, "X2": x2, "score": cb}


def run_vectorised(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Vectorised Stages 1-4 pipeline. Returns the estimates table."""
    dyn = _build_dyn(df)
    dyn = r2_prior_series(dyn, mode)

    n = len(dyn)
    x0 = dyn["x0"].to_numpy()
    x1 = np.zeros(n); x2 = np.zeros(n)
    xm = x0.copy()
    converged = np.zeros(n, dtype=bool)

    for _ in range(50):
        act = ~converged
        if not act.any():
            break
        idx = np.flatnonzero(act)
        if len(idx) != n:
            sub = dyn.iloc[idx]
            sol = solve_x1x2_vector(sub, xm[idx], max_iter=12)
            x1[idx] = sol["X1"]; x2[idx] = sol["X2"]
        else:
            sol = solve_x1x2_vector(dyn, xm, max_iter=12)
            x1, x2 = sol["X1"], sol["X2"]
        xm_new = x0 - x1
        conv = np.abs(xm_new - xm) < 1e-6
        xm = np.where(act & ~conv, xm_new, xm)
        converged |= conv & act
        if not (~converged).any():
            break

    f = _resid(x1, x2, dyn, x0 - x1)
    score = np.einsum("ij,ij->i", f, f)

    out = {
        "design_id": dyn["design_id"].to_numpy(),
        "R1_hot": dyn["r1_hot"].to_numpy(),
        "R2": dyn["r2_prior"].to_numpy(),
        "R3": dyn["r3_est"].to_numpy(),
        "X1": x1, "X2": x2,
        "X3": dyn["xlr"].to_numpy() - x1,
        "Xm": x0 - x1,
        "X0": x0, "XLR": dyn["xlr"].to_numpy(),
        "converged": converged,
        "solver_score": score,
        "s": dyn["s_fl"].to_numpy(),
    }
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Reference path (plan §5.4) and metric aggregation (plan §5.3)
# ---------------------------------------------------------------------------

def _row_case(row, mode: str) -> ec.MotorCase:
    conn = str(row["Connection"]).strip().upper()
    connection = "delta" if conn.startswith("D") else "Y"
    return ec.MotorCase(
        V_LL=float(row["v_ph"] * np.sqrt(3.0)),
        f=float(row["f"]), P=int(row["pole"]),
        n_FL=float(row["n_FL"]), P_out=float(row["hp"]),
        I_FL=float(row["i_fl_line"]),
        PF_FL=float(row["pf_fl"]),
        eta_FL=0.9,
        I_LR=float(row["i_lr_line"]),
        T_LR=float(row["t_lr_pu"]),
        T_BD=1.8,
        R1_cold=float(row["r1_cold_feed"]),
        I_0=float(row["i_0_line"]),
        P_0=float(row["p_0"]),
        J=2.0,
        connection=connection,
        T_ambient_C=T_AMB,
        temp_rise_C=T_RISE,
        R2_override=float(row["r2_gt"]) if mode == "ground-truth" else None,
    )


def run_reference(df: pd.DataFrame, indices: np.ndarray, mode: str) -> tuple[list[dict], float]:
    """Per-row scalar ``fit()`` on the subsample; returns rows + wall time (s)."""
    dyn = _build_dyn(df)
    rows = dyn.iloc[indices]
    t0 = time.perf_counter()
    results = []
    for _, r in rows.iterrows():
        case = _row_case(r, mode)
        try:
            est = ec.EquivalentCircuitEstimator(case).fit()
        except Exception:
            results.append({"design_id": int(r["design_id"]), "raised": True})
            continue
        est["design_id"] = int(r["design_id"])
        est["raised"] = False
        results.append(est)
    dt = time.perf_counter() - t0
    return results, dt


# ---------------------------------------------------------------------------
# Report + CLI
# ---------------------------------------------------------------------------

def _attach_ground_cols(res: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Copy ground-truth/serving columns onto the estimates for the report.

    ``res`` is produced row-by-row in the same order as ``df``, so columns are
    aligned by position (the dataset has no unique design id; ``design_id``
    carries the source row index). Ground-truth parameter columns get a
    ``gt_`` prefix so they never clobber the estimate columns of the same name.
    """
    res = res.copy()
    for c in ["Connection", "Voltage", "HorsePower", "Frequency",
              "RPM", "Amps", "NoLoadAmps", "WindingResistAt105"]:
        if c in df.columns:
            res[c] = df[c].to_numpy()
    res["Pole"] = df["PoleSpeed"].to_numpy()
    for p in METRIC_PARAMS:
        res[f"gt_{p}"] = df[p].to_numpy(dtype=float)
    return res


def compute_errors(res: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Relative errors for METRIC_PARAMS (plan §5.3), aligned by row order."""
    out = res.copy()
    for p in METRIC_PARAMS:
        gt_v = df[p].to_numpy(dtype=float)
        est_v = res[p].to_numpy(dtype=float)
        denom = np.where(np.abs(gt_v) > 1e-12, np.abs(gt_v), np.inf)
        out[f"err_{p}"] = (est_v - gt_v) / denom
    return out


def _mae(x):
    return float(np.mean(np.abs(x)))


def _rmse(x):
    return float(np.sqrt(np.mean(np.square(x))))


def _median(x):
    return float(np.median(x))


def _summaries(err: pd.DataFrame) -> pd.DataFrame:
    """Per-parameter MAE/RMSE/median/P10/P90/outlier summary (plan §5.3)."""
    rows = {}
    for p in METRIC_PARAMS:
        e = err[f"err_{p}"].to_numpy(dtype=float)
        n_out = int((np.abs(e) > OUTLIER_THRESHOLD).sum())
        rows[p] = {
            "MAE": _mae(e), "RMSE": _rmse(e), "Median": _median(e),
            "P10": float(np.percentile(e, 10)), "P90": float(np.percentile(e, 90)),
            "outliers": n_out, "outliers_pct": 100.0 * n_out / len(e),
        }
    return pd.DataFrame(rows)


def _group_tables(err: pd.DataFrame, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """MAE per parameter per subset group (plan §5.3)."""
    labels = {
        "connection": err["Connection"].astype(str).str.strip().str.upper(),
        "pole": err["Pole"].astype(str),
        "voltage": pd.cut(err["Voltage"].to_numpy(dtype=float),
                          VOLTAGE_EDGES, labels=VOLTAGE_LABELS),
        "hp": pd.cut(err["HorsePower"].to_numpy(dtype=float),
                     HP_EDGES, labels=HP_LABELS),
    }
    tables = {}
    for gname, g in labels.items():
        rows = {}
        for p in METRIC_PARAMS:
            col = err[f"err_{p}"]
            rows[p + "_MAE"] = col.groupby(g, observed=False).apply(_mae)
        tables[gname] = pd.DataFrame(rows)
    return tables


def _ref_table(ref_results: list[dict], res: pd.DataFrame) -> pd.DataFrame:
    """Max |rel diff| per parameter between vectorised and reference (§5.4)."""
    vec = res.set_index("design_id")
    rows = []
    for r in ref_results:
        if r.get("raised"):
            continue
        v = vec.loc[r["design_id"]]
        reld = {"design_id": int(r["design_id"])}
        for p in METRIC_PARAMS:
            a, b = float(r[p]), float(v[p])
            denom = max(abs(b), abs(a), 1e-12)
            reld[p] = abs(a - b) / denom
        rows.append(reld)
    return pd.DataFrame(rows)


def fl_consistency(df: pd.DataFrame):
    """Full-load consistency check: does the ground-truth circuit reproduce the
    nameplate full-load current? Returns (ratio, is_delta). Uses the
    wye-equivalent convention (V_ph = V_LL/sqrt(3), I_ph = I_line for both);
    ratio ~1 means the per-phase circuit and the nameplate point agree."""
    conn = _is_delta(df)
    j = np.where(conn, 3.0, 1.0)
    v_ph = df["Voltage"].to_numpy(dtype=float) / np.sqrt(3.0)
    i_ph = df["Amps"].to_numpy(dtype=float)
    ns = 120.0 * df["Frequency"].to_numpy(dtype=float) / df["PoleSpeed"].to_numpy(dtype=float)
    s = (ns - df["RPM"].to_numpy(dtype=float)) / ns
    r1 = df["WindingResistAt105"].to_numpy(dtype=float) / j
    zm = 1j * df["Xm"].to_numpy(dtype=float)
    z2 = df["R2"].to_numpy(dtype=float) / np.maximum(s, 1e-9) + 1j * df["X2"].to_numpy(dtype=float)
    zpar = zm * z2 / (zm + z2)
    z_fl = r1 + 1j * df["X1"].to_numpy(dtype=float) + zpar
    ratio = (v_ph / np.abs(z_fl)) / np.maximum(i_ph, 1e-12)
    return ratio, conn


def write_report(df: pd.DataFrame, results: dict, prep: dict) -> str:
    """Render and write ``validation_report.md``.

    ``results`` maps mode -> dict(res, ref_results, ref_time, t_vec_s).
    """
    lines = []
    w = lines.append
    w("# Validation Report (plan.md Stage 5)")
    w("")
    w(f"- Dataset          : `{RESULTS_CSV.name}`")
    w(f"- Rows (total)     : {prep['total']}")
    w(f"- Rows (valid)     : {prep['valid']} (dropped `slip<=0`: "
      f"{prep['n_slip']}, `R2/R3>=1`: {prep['n_ratio1']})")
    w(f"- Inputs           : real measurement columns `PowerFactor100`, "
      f"`LockedRotorAmps`, `NoLoadAmps`, `NoLoadWatts`, `WindingResistAt105` "
      f"(plan §5.2).")
    w("")
    w("**T_LR note:** `LockedRotorTorque` is a **percentage of full-load "
      "torque**, so `T_LR = LockedRotorTorque / 100` (pu). The `R3` back-solve "
      "is therefore a real measurement (catalog spec vs design value).")
    w("")
    w(f"- Converged        : {prep['n_conv']}/{prep['valid']} "
      f"({100*prep['n_conv']/max(prep['valid'],1):.1f}%)")
    w("")

    w("## 1. Summary — run modes (plan §5.2.2 / §2.5)")
    w("")
    w("| Param | prior RMSE | prior MAE | ground-truth RMSE | ground-truth MAE |")
    w("|---|---:|---:|---:|---:|")
    for p in METRIC_PARAMS:
        row = []
        for m in ("prior", "ground-truth"):
            err = compute_errors(results[m]["res"], df)
            s = _summaries(err)[p]
            row.append(f"{s['RMSE']:.4f}")
            row.append(f"{s['MAE']:.4f}")
        w(f"| {p} | {row[0]} | {row[1]} | {row[2]} | {row[3]} |")
    w("")
    w("- **prior**: `R2 = R3 * R2R3_MODEL_D` (production view; prior error "
      "included).")
    w("- **ground-truth**: `R2 = R2_dataset` — the §2.5 experiment that "
      "isolates the circuit-solve error (with an accurate `R2/R3`, how close "
      "is the recovered circuit to ground truth?).")
    w("")
    w("**§2.5 answer:** the ground-truth-mode errors above are the solver's "
      "own contribution. Where they are small, the current Stages 1-4 "
      "procedure is adequate and effort should go into the `R2/R3` prior; "
      "where large, the solver needs rework first.")
    w("")

    w("## 1b. Full-load consistency of the ground truth (data check)")
    w("")
    ratio, conn = fl_consistency(df)
    for label, m in (("wye", ~conn), ("delta", conn)):
        r = ratio[m]
        w(f"- **{label}**: `I_implied / I_nameplate` median "
          f"{np.median(r):.2f} (P10-P90 {np.percentile(r, 10):.2f}-"
          f"{np.percentile(r, 90):.2f}), n = {int(m.sum())}.")
    w("")
    w("A ratio of ~1 means the ground-truth circuit reproduces the nameplate "
      "full-load point (wye-equivalent convention: `V_ph = V_LL/sqrt(3)`, "
      "`I_ph = I_line`). Both connections should be ~1; a persistent offset "
      "indicates a per-phase convention mismatch between the circuit values "
      "and the full-load measurements.")
    w("")

    for m in MODES:
        r = results[m]
        err = compute_errors(r["res"], df)
        conv = err["converged"].to_numpy(dtype=bool)
        n_conv = int(conv.sum())
        cov = n_conv / len(err)
        status = "PASS" if cov >= COVERAGE_MIN else "FAIL"
        w(f"## 2. Mode: {m}")
        w("")
        w(f"- Converged        : {n_conv}/{len(err)} ({100*cov:.1f}%) → {status}")
        w(f"- Vectorised time  : {r['t_vec_s']*1000:.1f} ms (all rows)")
        if r["ref_time"]:
            n_ref = len(r["ref_results"] or [])
            w(f"- Reference time   : {r['ref_time']:.2f} s for {n_ref} rows "
              f"({r['ref_time']/max(n_ref,1)*1000:.2f} ms/row)")
        w("")
        w("### Parameter relative errors")
        w("")
        w("| Param | MAE | RMSE | Median | P10 | P90 | >15% outliers |")
        w("|---|---|---|---|---|---|---|")
        s = _summaries(err)
        for p in METRIC_PARAMS:
            row = s[p]
            w(f"| {p} | {row['MAE']:.4f} | {row['RMSE']:.4f} | {row['Median']:+.4f} | "
              f"{row['P10']:+.4f} | {row['P90']:+.4f} | {row['outliers_pct']:.1f}% "
              f"({row['outliers']}) |")
        w("")

        w("### Outlier rows (>15% rel err)")
        w("")
        for p in METRIC_PARAMS:
            e = err[f"err_{p}"].to_numpy(dtype=float)
            mask = np.abs(e) > OUTLIER_THRESHOLD
            if mask.sum() == 0:
                continue
            idx = np.argsort(-np.abs(e))[:8]
            ids = err["design_id"].to_numpy(dtype=int)
            det = ", ".join([f"{ids[i]}({e[i]:+.2f})" for i in idx])
            w(f"- **{p}**: {det}")
        w("")

        w("### By subset (MAE)")
        w("")
        groups = _group_tables(err, df)
        for gname in ["connection", "pole", "voltage", "hp"]:
            tbl = groups[gname]
            w(f"#### {gname.replace('_', ' ').title()}")
            w("")
            w("| Group | " + " | ".join([f"{p} MAE" for p in METRIC_PARAMS]) + " |")
            w("|" + "---|" * (1 + len(METRIC_PARAMS)))
            for label, row in tbl.iterrows():
                vals = " | ".join([f"{np.abs(row[p+'_MAE']):.4f}".replace("nan", "—")
                                   for p in METRIC_PARAMS])
                w(f"| {label} | {vals} |")
            w("")

    w("## 3. Reference cross-check (plan §5.4)")
    w("")
    for m in MODES:
        r = results[m]
        w(f"### {m}")
        w("")
        if r["ref_results"]:
            ref = _ref_table(r["ref_results"], r["res"])
            w("Per-row `EquivalentCircuitEstimator.fit()` vs vectorised path, "
              "max relative difference on each parameter.")
            w("")
            w("| Param | max |rel diff| | rows within 2% |")
            w("|---|---|---|")
            for p in METRIC_PARAMS:
                d = ref[p].to_numpy(dtype=float)
                within = float((d <= REFERENCE_TOL_REL).mean())
                w(f"| {p} | {d.max():.4e} | {100*within:.1f}% |")
            w("")
        else:
            w("_Per-row reference check skipped (re-run with `--subsample N`)._")
            w("")

    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def output_csv(mode: str) -> Path:
    return ROOT / "reports" / f"validation_results_{mode.replace('-', '_')}.csv"


def run_mode(df: pd.DataFrame, mode: str, subsample: int) -> dict:
    t0 = time.perf_counter()
    res = run_vectorised(df, mode)
    t_vec = time.perf_counter() - t0
    res["t_vec_s"] = t_vec
    res = _attach_ground_cols(res, df)
    res.to_csv(output_csv(mode), index=False)

    ref_results = None
    ref_time = None
    if subsample:
        ref_results, ref_time = run_reference(df, np.arange(min(subsample, len(df))), mode)
    return {"res": res, "ref_results": ref_results, "ref_time": ref_time,
            "t_vec_s": t_vec}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=MODES + ("both",), default="both",
                    help="run mode(s) (default both)")
    ap.add_argument("--subsample", type=int, default=200,
                    help="rows for per-row reference cross-check (0 disables)")
    ap.add_argument("--report-only", action="store_true",
                    help="recompute the report from saved validation_results_*.csv")
    args = ap.parse_args(argv)

    raw = pd.read_csv(RESULTS_CSV)
    if args.report_only:
        df = valid_rows(raw)
        results = {}
        for m in MODES:
            if not output_csv(m).exists():
                print(f"missing {output_csv(m)}; run without --report-only first")
                return 2
            res = pd.read_csv(output_csv(m))
            results[m] = {"res": res, "ref_results": None, "ref_time": None,
                          "t_vec_s": 0.0}
        prep = {"total": len(raw), "valid": len(df),
                "n_slip": int(((raw["SRPM"] - raw["RPM"]) / raw["SRPM"] <= 0).sum()),
                "n_ratio1": int((raw["R2"] / raw["R3"] >= 1.0).sum()),
                "n_conv": int((results["prior"]["res"]["converged"].astype(bool)).sum())}
        write_report(df, results, prep)
        return 0

    df = valid_rows(raw)
    prep = {
        "total": len(raw), "valid": len(df),
        "n_slip": int(((raw["SRPM"] - raw["RPM"]) / raw["SRPM"] <= 0).sum()),
        "n_ratio1": int((raw["R2"] / raw["R3"] >= 1.0).sum()),
        "n_conv": 0,
    }

    modes = MODES if args.mode == "both" else (args.mode,)
    results = {}
    for m in modes:
        print(f"running mode: {m}")
        results[m] = run_mode(df, m, args.subsample)
    prep["n_conv"] = int(results[modes[0]]["res"]["converged"].astype(bool).sum())

    write_report(df, results, prep)
    print(f"Wrote {REPORT_MD}")
    for m in modes:
        print(f"  {m}: wrote {output_csv(m)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

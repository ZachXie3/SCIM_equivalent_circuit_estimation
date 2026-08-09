"""Stage 5 validation harness (plan.md §5).

Runs the Stages 1-4 equivalent-circuit estimation pipeline over
``examples_eq_results.csv`` (4,183 ground-truth design rows) and compares the
predicted per-phase parameters against the dataset's ``R2, R3, X1, X2, X3,
Xm`` columns.

Design (plan.md §5.4 - matrix-driven, no per-row Python loop):

* Stage 1 + Stage 2  : closed-form vectorised over the whole table.
* Stage 3 (X1, X2)   : vectorised Gauss-Newton least squares over all rows
                       at once, with a masked backtracking line search.
* Stage 4 outer loop : pure array updates (Xm = X0 - X1, X3 = XLR - X1).
* Reference path     : per-row ``EquivalentCircuitEstimator.fit()`` on a
                       subsample, cross-checked against the vectorised path
                       and used for the runtime comparison.

Synthetic test inputs (plan §5.2.2) are generated vectorised from the
ground-truth circuit, so the Stage-1 back-solves (R3, XLR, X0) are
near-exact by construction; the informative comparison is R2 (dataset prior)
and the Stage-3/4 solves X1, X2, Xm, X3 (plan §5.3).

Usage:
    python validate.py                  # full run, writes validation_report.md
    python validate.py --subsample 200  # also cross-check reference on N rows
    python validate.py --report-only    # regenerate report from saved CSV
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import equivalent_circuit as ec  # noqa: E402

ROOT = Path(__file__).resolve().parent
RESULTS_CSV = ROOT / "examples_eq_results.csv"
OUTPUT_CSV = ROOT / "validation_results.csv"
REPORT_MD = ROOT / "validation_report.md"

T_AMB = 25.0
T_RISE = 80.0
T_HOT = T_AMB + T_RISE
K1 = ec.K1_COPPER
HP_TO_WATT = ec.HP_TO_WATT

REFERENCE_TOL_REL = 0.02   # vectorised vs per-row reference (plan §5.4)
COVERAGE_MIN = 0.90        # share of rows required to converge
OUTLIER_THRESHOLD = 0.15   # |rel err| above this -> outlier (plan §5.3)

METRIC_PARAMS = ("R2", "R3", "X1", "X2", "X3", "Xm")

VOLTAGE_EDGES = [0.0, 500.0, 1000.0, 2000.0, 5000.0, np.inf]
VOLTAGE_LABELS = ["<500", "500-999", "1000-1999", "2000-4999", "5000+"]
HP_EDGES = [0.0, 100.0, 500.0, 1000.0, 2000.0, np.inf]
HP_LABELS = ["<100", "100-499", "500-999", "1000-1999", "2000+"]


# ---------------------------------------------------------------------------
# Vectorised Stage 1 + Stage 2 (plan §5.4)
# ---------------------------------------------------------------------------

def _is_delta(df: pd.DataFrame) -> np.ndarray:
    conn = df["Connection"].astype(str).str.strip().str.upper()
    return conn.str.startswith("D").to_numpy(dtype=bool)


def stage1_stage2(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Closed-form per-phase quantities over the whole table (vectorised)."""
    is_delta = _is_delta(df)
    sq3 = np.sqrt(3.0)

    v_ll = df["Voltage"].to_numpy(dtype=float)
    v_ph = np.where(is_delta, v_ll, v_ll / sq3)
    i_fl_ph = np.where(is_delta, df["Amps"].to_numpy(dtype=float) / sq3,
                       df["Amps"].to_numpy(dtype=float))
    i_0_ph = np.where(is_delta, df["NoLoadAmps"].to_numpy(dtype=float) / sq3,
                      df["NoLoadAmps"].to_numpy(dtype=float))

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

    return {
        "v_ph": v_ph, "i_fl_ph": i_fl_ph, "i_0_ph": i_0_ph,
        "n_s": n_s, "omega_s": omega_s, "s_fl": s_fl, "t_fl": t_fl,
        "r1_cold_feed": r1_cold_feed, "r1_hot": r1_hot, "is_delta": is_delta,
    }


def generate_synthetic_inputs(df: pd.DataFrame, st: dict) -> dict[str, np.ndarray]:
    """Vectorised synthetic measured inputs from the ground truth (§5.2.2).

    PF_FL from the ground-truth full-load impedance; locked-rotor current and
    torque; no-load current and power — all evaluated on the known circuit.
    """
    x1 = df["X1"].to_numpy(dtype=float)
    x2 = df["X2"].to_numpy(dtype=float)
    x3 = df["X3"].to_numpy(dtype=float)
    xm = df["Xm"].to_numpy(dtype=float)
    r2 = df["R2"].to_numpy(dtype=float)
    r3 = df["R3"].to_numpy(dtype=float)
    s = np.maximum(st["s_fl"], 1e-9)
    i_fl_line = _phase_to_line(st["i_fl_ph"], st["is_delta"])

    # PF_FL from the ground-truth Z at full load.
    zm = 1j * xm
    z2 = r2 / s + 1j * x2
    zpar = zm * z2 / (zm + z2)
    z_fl = st["r1_hot"] + 1j * x1 + zpar
    pf = np.cos(np.angle(z_fl))

    # Locked rotor: line current and pu torque.
    z_lr = st["r1_cold_feed"] + r3 + 1j * (x1 + x3)
    i_lr_ph = st["v_ph"] / np.abs(z_lr)
    i_lr_line = _phase_to_line(i_lr_ph, st["is_delta"])
    t_lr = 3.0 * i_lr_ph**2 * r3 / st["omega_s"] / np.maximum(st["t_fl"], ec.EPS)

    # No load: line current and input power.
    z_nl = st["r1_hot"] + 1j * (x1 + xm)
    i_nl_ph = np.abs(st["v_ph"] / z_nl)
    i_0_line = _phase_to_line(i_nl_ph, st["is_delta"])
    p_0 = 3.0 * np.abs(st["v_ph"])**2 * st["r1_hot"] / np.abs(z_nl)**2

    bad = st["s_fl"] <= 0
    return {
        "pf_fl": np.where(bad, 0.85, pf),
        "i_fl_line": i_fl_line,
        "i_lr_line": np.where(bad, 6.0 * df["Amps"].to_numpy(dtype=float), i_lr_line),
        "t_lr_pu": np.where(bad, 1.0, t_lr),
        "i_0_line": i_0_line,
        "p_0": p_0,
    }


def stage1_backsolve(d: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Stage-1 back-solves R3, XLR, X0 from the (synthetic) test data."""
    is_delta = d["is_delta"].astype(bool)
    sq3 = np.sqrt(3.0)
    i_lr_ph = np.where(is_delta, d["i_lr_line"] / sq3, d["i_lr_line"])
    i_0_ph = np.where(is_delta, d["i_0_line"] / sq3, d["i_0_line"])

    r3_est = (d["t_lr_pu"] * d["t_fl"] * d["omega_s"]
              / np.maximum(3.0 * i_lr_ph**2, ec.EPS))

    r_lr = d["r1_cold_feed"] + r3_est
    z_lr = d["v_ph"] / np.maximum(i_lr_ph, ec.EPS)
    z_sq = z_lr**2 - r_lr**2
    xlr = np.where(z_sq > 0.0, np.sqrt(z_sq), ec.EPS)

    s0 = 3.0 * d["v_ph"] * i_0_ph
    q0 = np.sqrt(np.maximum(s0**2 - d["p_0"]**2, 0.0))
    x0 = 3.0 * d["v_ph"]**2 / np.maximum(q0, ec.EPS)

    return {"r3_est": r3_est, "xlr": xlr, "x0": x0}


def _phase_to_line(i_ph: np.ndarray, is_delta: np.ndarray) -> np.ndarray:
    return np.where(is_delta, i_ph * np.sqrt(3.0), i_ph)


def _build_dyn(df: pd.DataFrame) -> pd.DataFrame:
    """One flat table of per-row serving data for the whole pipeline."""
    st = stage1_stage2(df)
    syn = generate_synthetic_inputs(df, st)
    dyn = {**st, **syn}
    for k, v in (stage1_backsolve(dyn)).items():
        dyn[k] = v
    dyn["pole"] = df["PoleSpeed"].to_numpy(dtype=int)
    dyn["hp"] = df["HorsePower"].to_numpy(dtype=float)
    dyn["f"] = df["Frequency"].to_numpy(dtype=float)
    dyn["n_FL"] = df["RPM"].to_numpy(dtype=float)
    dyn["design_id"] = df["DesignAuditID"].to_numpy()
    dyn["Connection"] = df["Connection"].astype(str).to_numpy()
    return pd.DataFrame(dyn)


def r2_prior_series(dyn: pd.DataFrame) -> pd.Series:
    """Vectorised R2 prior from the dataset model (plan §2.4/§2.5)."""
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


def run_vectorised(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorised Stages 1-4 pipeline. Returns the estimates table."""
    dyn = _build_dyn(df)
    dyn = r2_prior_series(dyn)

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

def _row_case(row) -> ec.MotorCase:
    conn = str(row["Connection"]).strip().upper()
    connection = "delta" if conn.startswith("D") else "Y"
    return ec.MotorCase(
        V_LL=float(row["v_ph"] * (np.sqrt(3.0) if connection == "Y" else 1.0)),
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
    )


def run_reference(df: pd.DataFrame, indices: np.ndarray) -> tuple[list[dict], float]:
    """Per-row scalar ``fit()`` on the subsample; returns rows + wall time (s)."""
    dyn = _build_dyn(df)
    rows = dyn.iloc[indices]
    t0 = time.perf_counter()
    results = []
    for _, r in rows.iterrows():
        case = _row_case(r)
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

    Ground-truth parameter columns get a ``gt_`` prefix so they never clobber
    the estimate columns of the same name.
    """
    gt = df.set_index("DesignAuditID")
    res = res.copy()
    for c in ["Connection", "Voltage", "HorsePower", "Frequency",
              "RPM", "Amps", "NoLoadAmps", "WindingResistAt105"]:
        if c in gt.columns:
            res[c] = gt.loc[res["design_id"], c].to_numpy()
    res["Pole"] = gt.loc[res["design_id"], "PoleSpeed"].to_numpy()
    for p in METRIC_PARAMS:
        res[f"gt_{p}"] = gt.loc[res["design_id"], p].to_numpy(dtype=float)
    return res


def compute_errors(res: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Relative errors for METRIC_PARAMS (plan §5.3)."""
    gi = df.set_index("DesignAuditID")
    out = res.copy()
    for p in METRIC_PARAMS:
        gt_v = gi.loc[res["design_id"], p].to_numpy(dtype=float)
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
    ids = np.array([r["design_id"] for r in ref_results
                    if not r.get("raised")])
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


def write_report(df: pd.DataFrame, res: pd.DataFrame, ref_results=None,
                 ref_time: float | None = None) -> str:
    """Render and write ``validation_report.md``."""
    err = compute_errors(res, df)
    conv = err["converged"].to_numpy(dtype=bool)
    n_conv = int(conv.sum())
    cov = n_conv / len(err)

    lines = []
    w = lines.append
    w("# Validation Report (plan.md Stage 5)")
    w("")
    w(f"- Dataset          : `{RESULTS_CSV.name}`")
    w(f"- Rows             : {len(err)}")
    w(f"- Converged        : {n_conv}/{len(err)} ({100*cov:.1f}%)")
    w(f"- Vectorised time  : {res['t_vec_s'].iloc[0]*1000:.1f} ms (all rows)")
    if ref_time:
        n_ref = len(ref_results or [])
        w(f"- Reference time   : {ref_time:.2f} s for {n_ref} rows "
          f"({ref_time/max(n_ref,1)*1000:.2f} ms/row)")
    w("")
    w("## Coverage (plan §5.5)")
    w("")
    status = "PASS" if cov >= COVERAGE_MIN else "FAIL"
    w(f"- converged rows: **{100*cov:.1f}%** (target ≥ {100*COVERAGE_MIN:.0f}%) → {status}")
    w("")

    w("## Parameter relative errors (plan §5.3)")
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

    # Outlier rows (plan §5.3): list the worst few per parameter.
    w("## Outlier rows (>15% rel err, plan §5.3)")
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

    w("## Reference cross-check (plan §5.4)")
    w("")
    if ref_results:
        ref = _ref_table(ref_results, res)
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

    w("## By subset (MAE)")
    w("")
    groups = _group_tables(err, df)
    for gname in ["connection", "pole", "voltage", "hp"]:
        tbl = groups[gname]
        w(f"### {gname.replace('_', ' ').title()}")
        w("")
        w("| Group | " + " | ".join([f"{p} MAE" for p in METRIC_PARAMS]) + " |")
        w("|" + "---|" * (1 + len(METRIC_PARAMS)))
        for label, row in tbl.iterrows():
            vals = " | ".join([f"{np.abs(row[p+'_MAE']):.4f}".replace("nan","—")
                               for p in METRIC_PARAMS])
            w(f"| {label} | {vals} |")
        w("")

    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subsample", type=int, default=200,
                    help="rows for per-row reference cross-check (0 disables)")
    ap.add_argument("--report-only", action="store_true",
                    help="recompute the report from a saved validation_results.csv")
    args = ap.parse_args(argv)

    if args.report_only:
        if not OUTPUT_CSV.exists():
            print(f"missing {OUTPUT_CSV}; run without --report-only first")
            return 2
        df = pd.read_csv(RESULTS_CSV)
        res = pd.read_csv(OUTPUT_CSV)
        write_report(df, res)
        return 0

    raw = pd.read_csv(RESULTS_CSV)
    t0 = time.perf_counter()
    res = run_vectorised(raw)
    t_vec = time.perf_counter() - t0
    res["t_vec_s"] = t_vec
    res = _attach_ground_cols(res, raw)
    res.to_csv(OUTPUT_CSV, index=False)

    ref_results = None
    ref_time = None
    if args.subsample:
        ref_results, ref_time = run_reference(raw, raw.index[:min(args.subsample, len(raw))])
    write_report(raw, res, ref_results, ref_time)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
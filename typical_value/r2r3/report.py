"""Generate ``reports/r2r3_report.md`` — dataset-driven R2/R3 model study.

Builds on ``models.py`` (plan.md Stage 2.4) and ``constant_bin.py`` (the
approved constant-per-bin table, plan.md §2.7 / §11.1): fits Models A-D on
``data/eq_parameters.csv``, scores predicted R2, selects the simplest model
capturing most of the available accuracy, and documents the fitted priors
for use by the estimator (plan.md Stage 2.5). Sections 9-11 cover the factor
review, the rotor-slot diagnostic, and the community-facing nameplate-only
methods.

Usage:
    python r2r3/report.py [output_markdown]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import constant_bin as cb
from . import data as data_mod
from . import models as m

DEFAULT_OUTPUT_MD = Path(__file__).resolve().parent.parent / "reports" / "r2r3_report.md"


def model_metrics_rows() -> tuple[list[str], dict, str]:
    """Return (markdown table rows, scores dict, selected model name)."""
    df = data_mod.valid_rows(data_mod.add_slip(data_mod.load_results()))
    rows, selected = m.score_table(df)
    scores = {r["name"]: r for r in rows}
    lines = []
    for r in rows:
        mark = " (selected)" if r["name"] == selected else ""
        lines.append(
            f"| {r['name']}{mark} | {r['MAE_ohm']:.4f} | {r['RMSE_ohm']:.4f} | "
            f"{r['MAE_rel']*100:.2f}% | {r['RMSE_rel']*100:.2f}% | "
            f"{r['P10_rel']*100:.2f}% | {r['P50_rel']*100:.2f}% | {r['P90_rel']*100:.2f}% |"
        )
    return lines, scores, selected


def model_d_table(params: dict) -> list:
    lines = []
    for pole in sorted(params):
        p = params[pole]
        ratio = np.exp(p["b"]) * 0.01**p["a"] * 1.0**p["c"]
        lines.append(
            f"| {pole} | {p['a']:.3f} | {p['c']:.3f} | {p['b']:.3f} | "
            f"`{np.exp(p['b']):.4g} * s_fl^{p['a']:.4f} * HP^{p['c']:.4f}` | "
            f"{ratio:.3f} |"
        )
    return lines


def _design_matrix_d(df: pd.DataFrame) -> np.ndarray:
    """Per-pole Model D design matrix: [1, ln s_fl, ln HP] per pole count."""
    cols = []
    for p in sorted(df["PoleSpeed"].unique()):
        m_ = (df["PoleSpeed"] == p).to_numpy()
        cols.append(
            np.column_stack([m_, m_ * np.log(df["s_fl"].to_numpy()),
                             m_ * np.log(df["HorsePower"].to_numpy())])
        )
    return np.hstack(cols)


def _rmse_rel(e: np.ndarray) -> float:
    return float(100.0 * np.sqrt(np.mean(e**2)))


def _dummy_matrix(df: pd.DataFrame, col: str, levels: list[str]) -> np.ndarray:
    return pd.get_dummies(df[col].astype(str), dtype=float) \
        .reindex(columns=levels, fill_value=0).to_numpy()


def _factor_eta2(resid: np.ndarray, group: pd.Series) -> float:
    s = pd.Series(resid, index=group.index)
    means = s.groupby(group, observed=True).mean()
    counts = s.groupby(group, observed=True).size()
    ss_b = float(np.sum(counts * (means - resid.mean()) ** 2))
    ss_t = float(np.sum((resid - resid.mean()) ** 2))
    return ss_b / ss_t if ss_t > 0 else 0.0


def _rmse_with_extra(X: np.ndarray, y: np.ndarray, extra: np.ndarray) -> float:
    extra = np.asarray(extra, dtype=float)
    if extra.ndim == 1:
        extra = extra[:, None]
    b, *_ = np.linalg.lstsq(np.hstack([X, extra]), y, rcond=None)
    return _rmse_rel(y - np.hstack([X, extra]) @ b)


def factor_review_section(df: pd.DataFrame) -> list[str]:
    """Section 9 — can R2/R3 be treated as a constant? (nameplate factors)."""
    y = np.log(df["R2"] / df["R3"]).to_numpy()
    X = _design_matrix_d(df)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ b
    base = _rmse_rel(resid)

    slot = _factor_eta2(resid, df["RotorSlotType"].astype(str))
    mat = _factor_eta2(resid, df["BarMaterial"].astype(str))
    kva = _factor_eta2(resid, df["NemaKVACode"].astype(str))
    name = _factor_eta2(resid, df["Name"].astype(str))
    des = _factor_eta2(resid, df["NemaDesign"].astype(str))
    pole = _factor_eta2(resid, df["PoleSpeed"].astype(str))

    def rmse_with(col: str, levels: list[str]) -> float:
        return _rmse_with_extra(X, y, _dummy_matrix(df, col, levels))

    r_slot = rmse_with("RotorSlotType", ["Dbl - Closed", "Diamond", "Trapeze", "Rectang", "Oval"])
    r_mat = rmse_with("BarMaterial", ["AL", "CU", "Special"])
    r_kva = rmse_with("NemaKVACode", ["D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "C", "V", "P", "R"])
    r_name = rmse_with("Name", ["TIC Premium Brand", "Standard / High Efficiency", "TIC EPACT", "NEMA Premium4 Brand"])
    r_des = rmse_with("NemaDesign", ["A", "A,B", "A,B,C", "C"])

    lr = np.log((df["LockedRotorAmps"] / df["Amps"]).to_numpy())
    r_lr = _rmse_with_extra(X, y, lr)
    r_lr_partial = float(np.corrcoef(resid, lr)[0, 1])

    return [
        "",
        "## 9. Factor review — can R2/R3 be treated as a constant?",
        "",
        "Each factor is scored against the fitted Model D residual "
        "(per-pole `s_fl^a * HP^c`, baseline RMSE "
        f"{base:.1f}%). For categorical factors, `eta^2` = share of the "
        "unexplained variance the factor captures; RMSE is the relative-error "
        "RMSE after adding the factor to Model D.",
        "",
        "| Factor | eta^2 (vs Model D resid) | RMSE with factor | verdict |",
        "|---|---|---|---|",
        f"| Rotor bar geometry (`RotorSlotType`) | {slot:.3f} | {r_slot:.1f}% | strong - see section 10 |",
        f"| Rotor material (`BarMaterial` = `ERMaterial`) | {mat:.3f} | {r_mat:.1f}% | strong, 1:1 with slot shape |",
        f"| `NemaKVACode` | {kva:.3f} | {r_kva:.1f}% | moderate; coarse bin of locked-rotor current |",
        f"| `NemaDesign` | {des:.3f} | {r_des:.1f}% | weak; Design C +25% but n = 9 |",
        f"| `Name` (efficiency level) | {name:.3f} | {r_name:.1f}% | weak; mostly material leakage |",
        f"| Pole count | {pole:.3f} | - | already absorbed by per-pole coefficients |",
        "| `FrameSize` (numeric) | - | r = 0.07 | none |",
        f"| `LockedRotorAmps`/`Amps` | - | partial r = {r_lr_partial:.2f}; {r_lr:.1f}% | weak; mostly a pole proxy |",
        "",
        "Notes per factor:",
        "",
        "- **Name (efficiency level): no.** `Standard / High Efficiency` runs ~+7% "
        "above `TIC Premium Brand`, but within-group spread is +-22-29% and the "
        "offset is mostly the higher copper-bar share in the Standard group. "
        "`NEMA Premium4 Brand` has only 8 rows.",
        "- **BarMaterial / ERMaterial: identical columns in every row.** Copper bars "
        "shift the ratio strongly (~+46% in log space at fixed slip/pole/HP), yet the "
        "raw geometric-mean ratio is lower for copper (deeper skin effect): 2-pole AL "
        "0.313 vs CU 0.229. Material matters, but it is 1:1 with slot shape "
        "(section 10), the cleaner descriptor.",
        "- **Pole count:** a constant per pole is useless (RMSE 67-100%); slip must "
        "stay in the formula. Pole is already captured by Model D's per-pole "
        "coefficients.",
        "- **NemaKVACode:** monotonic - code G -5.9% ... code N +20% vs Model D. "
        "It is only a discrete proxy for locked-rotor current.",
        "- **NemaDesign:** no dependable signal (Design C +25% but n = 9; "
        "1040 rows have no design code).",
        "- **FrameSize:** no signal (numeric r = 0.07, log r = -0.02); the frame code "
        "is not a dependable proxy.",
        "- **LockedRotorAmps/Amps:** not linear (raw r = 0.07, log-log r = 0.19; "
        f"partial r = {r_lr_partial:.2f} after slip/pole/HP). Mostly a pole proxy: "
        "8-pole motors lock at ~4-5x full load, 2-pole at ~7-8x.",
    ]


def slot_section(df: pd.DataFrame) -> list[str]:
    """Section 10 — rotor slot shape explains the residual (internal diagnostic).

    Kept for engineering context only: slot geometry is NOT nameplate-available,
    so it is not part of the end-user method in section 11.
    """
    df = df.copy()
    df["ratio"] = df["R2"] / df["R3"]
    lines = [
        "",
        "## 10. Rotor slot shape — internal diagnostic (NOT an end-user input)",
        "",
        "For engineering context: `RotorSlotType` explains most of the Model D "
        "residual, and is perfectly confounded with material (AL -> "
        "Dbl - Closed / Diamond / Oval; CU / Special -> Rectang / Trapeze). It is "
        "**not** part of the recommended method below because slot geometry is not "
        "available from the nameplate.",
        "",
        "| RotorSlotType | Material | n | geo-mean R2/R3 | std (log) | slip exponent |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for st, g in df.groupby("RotorSlotType", sort=False):
        b, *_ = np.linalg.lstsq(
            np.column_stack([np.ones(len(g)), np.log(g["s_fl"])]),
            np.log(g["ratio"]), rcond=None,
        )
        gm = float(np.exp(np.log(g["ratio"]).mean()))
        lines.append(
            f"| {st} | {g['BarMaterial'].mode()[0]} | {len(g)} | {gm:.4f} | "
            f"{np.log(g['ratio']).std():.4f} | {b[1]:+.3f} |"
        )
    lines += [
        "",
        "- **Diamond and Oval** (shallow, low-height bars): the design tool produces "
        "`R2 = R3` exactly (std = 0) - negligible skin effect at these bar heights. "
        "The ratio is 1.0.",
        "- **Trapeze and Rectang** (deep bars): the ratio is essentially a fixed "
        "geometry constant (~0.23 / ~0.32) with almost no slip dependence.",
        "- **Dbl - Closed** (deep closed / double-cage): strongly slip-dependent.",
        "- Because an end user does not know the slot shape, the mixed population "
        "sets the achievable accuracy of any nameplate-only method (section 11).",
    ]
    return lines


def simple_method_section(df: pd.DataFrame) -> list[str]:
    """Section 11 — recommended simple, nameplate-only method."""
    df = df.copy()
    df["_ratio"] = df["R2"] / df["R3"]
    y = np.log(df["_ratio"]).to_numpy()

    def rel_rmse(pred_log: np.ndarray) -> float:
        # same relative-error convention as models.score_model:
        # rel = (pred_R2 - R2) / R2 = exp(pred_log - y) - 1
        return float(100.0 * np.sqrt(np.mean((np.exp(pred_log - y) - 1.0) ** 2)))

    # A: global constant
    c = float(y.mean())
    r_a = rel_rmse(np.full_like(y, c))
    # B: global slip power law
    Xb = np.column_stack([np.ones(len(df)), np.log(df["s_fl"])])
    bb, *_ = np.linalg.lstsq(Xb, y, rcond=None)
    r_b = rel_rmse(Xb @ bb)
    # B+HP: global slip + horsepower power law (recommended)
    X = np.column_stack([np.ones(len(df)), np.log(df["s_fl"]), np.log(df["HorsePower"])])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r_bh = rel_rmse(X @ b)
    # Model D (per-pole, reference)
    cols = []
    for p in sorted(df["PoleSpeed"].unique()):
        m_ = (df["PoleSpeed"] == p).to_numpy()
        cols.append(np.column_stack([m_, m_ * np.log(df["s_fl"]),
                                     m_ * np.log(df["HorsePower"])]))
    Xd = np.hstack(cols)
    bd, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    r_d = rel_rmse(Xd @ bd)

    rel = np.exp(X @ b - y) - 1.0
    p10, p50, p90 = (np.percentile(rel, q) * 100.0 for q in (10, 50, 90))

    lines = [
        "",
        "## 11. Recommended simple method — nameplate-only",
        "",
        "Only nameplate-available quantities are used: full-load slip "
        "`s_fl = (n_s - RPM)/n_s` and rated `HorsePower`. No pole grouping, no "
        "rotor material, no slot geometry (section 10).",
        "",
        "```text",
        f"R2/R3 = {np.exp(b[0]):.1f} * s_fl^{b[1]:.2f} * HorsePower^{b[2]:.2f}",
        "",
        f"      = {np.exp(b[0]):.1f} * s_fl^{b[1]:.2f} / HorsePower^{abs(b[2]):.2f}",
        "```",
        "",
        "Fit on the dataset (in-sample; predicted vs dataset R2):",
        "",
        "| Model | Predictors | RMSE (%) |",
        "|---|---|---|",
        "| A | constant | " f"{r_a:.1f} |",
        "| B | s_fl | " f"{r_b:.1f} |",
        "| B+HP (recommended) | s_fl, HorsePower | " f"{r_bh:.1f} |",
        "| D (reference) | s_fl, pole, HP | " f"{r_d:.1f} |",
        "",
        f"The single global formula is within ~{abs(r_bh - r_d):.1f} pp of Model D "
        "with three coefficients and no grouping, so it is preferred for "
        "community use.",
        "",
        f"Error bands (relative error on R2): P10-P50-P90 = "
        f"{p10:+.0f}% / {p50:+.0f}% / {p90:+.0f}%  (1-sigma ~+-25%).",
        "",
        "- Physical bound `R3 >= R2`: the formula stays below 1.0 for the entire "
        "dataset slip range (`s_fl` up to ~0.18); clamp the result to 1.0 as a "
        "safety (the estimator already does this).",
        "- If only slip is trusted, the fallback is "
        f"`R2/R3 = {np.exp(bb[0]):.1f} * s_fl^{bb[1]:.2f}` (RMSE {r_b:.1f}%).",
        "",
        "### 11.1 Approved table method (slip segments)",
        "",
        "A lookup-table alternative approved for community use: one constant "
        "`R2/R3` per slip band (computed by `r2r3/constant_bin.py`). No exponent "
        "math. For `s_fl >= 5%` the ratio is taken as **1.0** (extrapolating the "
        "upward trend).",
        "",
        "| Slip band | n (cases) | R2/R3 | Bin RMSE (%) | Bin P10-P90 (%) |",
        "|---|---|---:|---:|---:|",
    ]

    # --- approved slip-segment table (constant ratio per band) ------------
    score = cb.score_slip_segments(df)
    for band in score["bands"]:
        lines.append(
            f"| {band['label']} | {band['n']} | {band['ratio']:.3f} | "
            f"{band['RMSE_rel']:.1f} | {band['P10_rel']:+.1f} / {band['P90_rel']:+.1f} |"
        )

    lines += [
        "",
        f"Overall: RMSE = {score['RMSE_rel']:.1f}%, MAE = {score['MAE_rel']:.1f}%, "
        f"P10-P50-P90 = {score['P10_rel']:+.1f}% / {score['P50_rel']:+.1f}% / "
        f"{score['P90_rel']:+.1f}% (n = {score['n']}).",
        "",
        "- The table is as accurate as the slip-only power law "
        f"({score['RMSE_rel']:.0f}% vs {r_b:.0f}%) but ~7 pp behind the continuous "
        "slip+HP formula; use the formula when precision matters and the table "
        "when a hand lookup is preferred.",
        "- `s_fl >= 5%` has only 12 rows; the 1.0 entry is a trend extrapolation, "
        "not a fitted value.",
    ]
    return lines


def build_markdown() -> str:
    df_full = data_mod.add_slip(data_mod.load_results())
    df = data_mod.valid_rows(df_full)
    n_total, n_valid = len(df_full), len(df)

    metrics_lines, scores, selected = model_metrics_rows()
    params_c = m.fit_model_c(df)
    params_d = m.fit_model_d(df)
    grouping = cb.grouping_score_table(df)

    viol = df_full[df_full["R2"] > df_full["R3"]]
    n_viol = len(viol)
    max_ratio = viol["R2"].div(viol["R3"]).max() if n_viol else float("nan")

    best_rmse = min(r["RMSE_rel"] for r in scores.values())
    gain_over_c = (scores["C"]["RMSE_rel"] - scores["D"]["RMSE_rel"]) * 100

    lines = [
        "# R2/R3 Dataset-Driven Model (plan.md Stage 2.4)",
        "",
        "> Auto-generated by `typical_value/r2r3/report.py`.",
        "",
        "## 1. Objective",
        "",
        "Estimate the running/standstill rotor resistance ratio `R2/R3` as an empirical "
        "function of nameplate-available quantities, so `R2` can be derived from the "
        "measured locked-rotor resistance `R3`. The exact skin-effect formula cannot be "
        "used because it requires bar depth / material properties (`h`, `mu`, `sigma`) "
        "not available to the user (plan.md Stage 2.2).",
        "",
        "## 2. Models",
        "",
        "Power-law priors fitted in log-log space:",
        "",
        "| Model | Predictors |",
        "|---|---|",
        "| A | constant |",
        "| B | `s_fl` |",
        "| C | `s_fl`, pole count |",
        "| D | `s_fl`, pole count, HP |",
        "",
        "Enclosure-based variants (frame-size prefix / full frame size) were dropped by "
        "review — the frame-size code is not a dependable enclosure proxy.",
        "",
        "## 3. Data preparation",
        "",
        f"- Source: `typical_value/data/eq_parameters.csv` ({n_total} rows).",
        f"- Full-load slip computed from nameplate: `s_fl = (n_s - RPM)/n_s`, "
        f"`n_s = 120*f/poles`.",
        f"- Rows excluded from fitting: `s_fl <= 0` or `HP <= 0` ({n_total - n_valid}).",
        f"- Valid rows used: **{n_valid}**.",
        f"- `R3 >= R2` violations: {n_viol} rows (all floating-point ties at "
        f"ratio = 1.000000, max = {max_ratio:.6f}; no genuine R2 > R3 case).",
        "",
        "## 4. Model scoring",
        "",
        "Each model predicts `R2 = F(...) * R3`; metrics on predicted vs dataset R2.",
        "",
        "| Model | MAE (ohm) | RMSE (ohm) | MAE (%) | RMSE (%) | P10 (%) | P50 (%) | P90 (%) |",
        "|---|---|---|---|---|---|---|---|",
        *metrics_lines,
        "",
        f"Best RMSE_rel: {best_rmse*100:.2f}%. Selected model **{selected}** — the "
        "simplest model whose RMSE is within the 5% tolerance of the best "
        "(plan.md Stage 2.4).",
        "",
    ]

    if selected == "D":
        lines += [
            "## 5. Fitted parameters — Model D (per pole count)",
            "",
            "| Poles | a | c (HP exp) | b | R2/R3 = | R2/R3 @ s_fl=1%, HP=1 |",
            "|---|---|---|---|---|---|",
            *model_d_table(params_d),
            "",
        ]
        lines += [
            "Model C coefficients for reference: ",
            f"`{( {int(p): {k: round(v, 4) for k, v in d.items()} for p, d in params_c.items()} )}`.",
            "",
        ]
    else:
        lines += [
            "## 5. Fitted parameters",
            "",
            f"Selected model {selected} uses the global slip power law; coefficients: see "
            "`r2r3/models.py::fit_model_b`. (Model D per-pole coefficients available via "
            "`r2r3/models.py::fit_model_d`.)",
            "",
        ]

    lines += [
        "## 6. Grouping-constant exploration (saved for reference)",
        "",
        "Alternative formulation: a **constant** geometric-mean `R2/R3` per group "
        "(no slip dependence). In-sample error vs a slip-free grouping:",
        "",
        "| Grouping | Groups | RMSE (%) | MAE (%) |",
        "|---|---|---|---|",
    ]
    lines += [
        f"| {g['label']} | {g['n_groups']} | {g['RMSE_rel']*100:.2f}% | {g['MAE_rel']*100:.2f}% |"
        for g in grouping
    ]
    lines += [
        "",
        "Groupings without slip (voltage / pole / HP bucket / combos) lose badly to any "
        "slip-based model (A–D). Slip is the dominant predictor; HP adds value on top of "
        "slip. A slip-bin grouped table matches Model D only with ~100+ entries, so the "
        "power-law prior is preferred.",
        "",
        "## 7. Use in the estimator (plan.md Stage 2.5)",
        "",
        "- Prior: `R2_prior = R3 * exp(b) * s_fl^a * HP^c` using the selected model's "
        "coefficients.",
        f"- Prior spread: `RMSE_rel = {scores[selected]['RMSE_rel']*100:.1f}%`; P10–P90 "
        f"= {scores[selected]['P10_rel']*100:.1f}% to "
        f"{scores[selected]['P90_rel']*100:.1f}%.",
        "- This feeds `R2_prior` and `sigma_R2` in the long-term estimator architecture "
        "(`r_R2 = log(R2/R2_prior)/sigma_R2`).",
        "",
        "## 8. Engineering notes",
        "",
        "- Slip is the dominant predictor: Model B cuts RMSE from ~72% (Model A) to "
        "~33%.",
        f"- Modelling with HP: Model D gains `{gain_over_c:.1f} pp` RMSE over Model C.",
        "- In-sample scoring only; a train/test split is recommended once the prior is "
        "integrated into the estimator.",
        "",
    ]
    lines += factor_review_section(df)
    lines += slot_section(df)
    lines += simple_method_section(df)
    return "\n".join(lines)


def write_report(path: str | Path = DEFAULT_OUTPUT_MD) -> None:
    Path(path).write_text(build_markdown(), encoding="utf-8")
    print(f"Wrote {path}")


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUTPUT_MD
    write_report(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

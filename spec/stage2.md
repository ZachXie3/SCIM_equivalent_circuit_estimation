# Plan — SCIM Equivalent Circuit Estimation

## High-level goal

Estimate the single-cage per-phase equivalent-circuit parameters (R1, R2, X1, X2, X3, Xm) of a
three-phase squirrel-cage induction motor from **typical nameplate / datasheet values** only
(voltage, freq, poles, speed, HP, amps, PF, eta, no-load & locked-rotor test data — see
`spec/equivalent_circuit_estimation.md`).

## The problem

Given only nameplate/datasheet data, several parameters **cannot be split exactly** from first
principles. In particular:

- **X1 vs X2** — total leakage reactance must be split between stator and rotor.
- **R2 vs R3** — running rotor resistance vs standstill rotor resistance (skin effect).
- **Xm** — the no-load test yields the total no-load reactance `X0 = X1_true + Xm_true`; the X1
  component is not separable from the no-load test alone (`Xm = X0 − X1` once the split is known).

## The approach

Collect the **typical splits / ratios** from historical, fully-specified designs (the
`typical_value` dataset, which comes from the design tool's slot-by-slot reactance
calculations), then **reuse those ratios** as priors/constraints when estimating a new motor from
nameplate data only.

Current stage: **collect and validate the typical equivalent-circuit dataset.**

---

## Stage 1 — Collect typical equivalent circuit (current)

Deliverables in `typical_value/`:

| File | Role |
|---|---|
| `examples_input.csv` | 4,183 raw design rows (slot reactances, bar/end-ring R, nameplate). |
| `examples_calculator.py` | Calculates per-phase R1, R2, R3, X1, X2, X3, Xm, time constants per row (mirrors `EquivalentCircuitParameters Rev.5.xlsx`). |
| `examples_eq_results.csv` | 4,183 rows = input + calculator outputs. |
| `test/` | Regression tests + reference workbooks. |

### Task 1.1 — Review the results CSV for correctness

- Confirm all 4,183 rows produce physically plausible, finite outputs.
- Sanity-check derived quantities vs the reference workbook (`EquivalentCircuitParameters Rev.5.xlsx`).
- Validate ranges: R, X > 0; Xm >> X1, X2; time constants within expected bands; no NaN/Inf/0.
- Flag any rows with negative or degenerate values.

### Task 1.2 — Add data-integrity tests

So future changes to `examples_calculator.py` (or the input CSV) cannot silently break the data:

- Column set: input columns + output columns must match expected headers exactly.
- Row-count / ID uniqueness: 4,183 rows, unique `DesignAuditID`.
- Determinism: re-running the calculator reproduces `examples_eq_results.csv`.
- Physical bounds (per row, or as quantile checks across the dataset).
- Keep the existing 734250BQ Rev.5 regression test passing.

### Task 1.3 — Consistent variable naming across code & docs

- `equivalent_circuit_estimation.md` uses: `R1`, `R2`, `R3`, `X1`, `X2`, `X3`, `Xm`, `R1_cold`, `R1_hot`, `XLR`, `Xtot`, `alpha`.
- `examples_calculator.py` / results CSV use: `R1`, `R2`, `R3`, `X1`, `X2`, `X3`, `Xm`, `Zbase`, ...
- Align the naming (units, hot vs raw resistance, Ω vs pu, subscripts) across `equivalent_circuit.py`,
  `examples_calculator.py`, the CSVs, and the spec. Document the mapping table.

### Task 1.4 — Line↔phase conversion for ALL voltages & currents (FL, LR, NL)

`equivalent_circuit_estimation.md` (§A1) currently converts line→phase only for the
**full-load voltage** (`V_ph`), depending on connection (Y vs Δ/Delta). This is not enough —
the per-phase equivalent circuit requires every per-phase quantity, so the same conversion must
be applied consistently to:

- **Voltages:** `V_LL` → `V_ph` for all operating points (FL, LR, NL).
- **Currents:** `I_FL`, `I_LR`, `I_0` — line current vs phase current (`I_ph = I_L / √3` for
  Δ/Delta, `I_ph = I_L` for Y).
- **Powers:** verify the 3-phase vs per-phase power formulas (`P_0`, `Pin`) are consistent with
  the chosen voltage/current basis.

Update `equivalent_circuit_estimation.md` (Part A) and the affected formulas in `equivalent_circuit.py`.
(No code change now — tracked as a to-do; will be implemented as part of the estimation work.)

### Task 1.5 — Review estimation logic: known vs unknown (sets target for 1.6)

Go through `spec/equivalent_circuit_estimation.md` and classify each parameter as
**clearly known** (computable directly from nameplate/test data) vs **not known** (needs a
typical-value assumption). Examples already identified:

- **Clearly known:** `V_ph`, `n_s`, `s_FL`, `R1_hot`, `T_FL`, `R3` (Part A).
- **Known with caveats:** `Xm` (B1 — no-load gives `X0`, then `Xm = X0 − X1` once the split is known), `Xtot` (B2 — overestimated by the shunt-correction seed), `XLR`.
- **Not known / needs a prior:** the **X1 vs X2 split** (`alpha`), and the **R2/R1 ratio**
  (running rotor resistance relative to stator; skin-effect bridge `R2 ↔ R3` is TBD).

The output of this task is the **target list for 1.6**: which splits/ratios the dataset must
provide typical values for (currently: `alpha = X1/(X1+X2)`, and `R2/R1`; possibly others that
surface during review, e.g. `X3/X2`).

### Task 1.6 — Preliminary pattern analysis

Ask the dataset what "typical" ratios look like. Checks to run on `examples_eq_results.csv`,
focused on the targets identified in 1.5:

- **R1 / R2 by efficiency band** — do premium-efficiency (EfficiencyBand=20) motors have a
  similar R1/R2 ratio to standard (EfficiencyBand=10)?
- **R1 / R2 by pole count** (PoleSpeed 2 / 4 / 6 / 8) — does the ratio cluster by pole count?
- **X1 vs X2** — typical X1/X2 split (alpha = X1/(X1+X2)); does it vary by pole count / efficiency /
  power class?
- **Other checks**: X1/Xtot and X2/Xtot by pole; R2/R1 by power; X3 vs X2 (standstill split);
  Xm/Xtot; how the spread looks within each group (mean, median, std, P10–P90).
- Deliverable: a short analysis script + a summary of ratios (and their spread) to seed priors.

---

## Stage 2 — (planned, not started)

- Feed the collected typical ratios as priors into `equivalent_circuit.py` (e.g. alpha prior,
  R2/R1 prior, Xm iteration).
- Resolve the known unknowns in `spec/equivalent_circuit_estimation.md` where the dataset
  provides constraints.


---

## Additional Engineering Direction — Dataset-Driven Priors and Adaptive Solver

### R2/R3 model development roadmap

Evaluate progressively more sophisticated dataset-derived models:

```text
Model A:
R2/R3 = constant
```

Group by:

```text
pole count
```

Then evaluate:

```text
Model B:
R2/R3 = f(s_fl)
```

Then:

```text
Model C:
R2/R3 = f(s_fl, pole_count)
```

Then:

```text
Model D:
R2/R3 = f(s_fl, pole_count, efficiency_band)
```

For each model:

```text
Predict R2
Compare predicted R2 to dataset R2
Compute MAE, RMSE, and P10/P50/P90 error bands
```

Select the simplest model that captures most of the available accuracy.

Engineering expectation:

```text
Slip is likely the strongest predictor.
Pole count is likely the second strongest predictor.
Horsepower may provide smaller incremental improvement.
```

These assumptions must be validated using the historical dataset.

### Long-term estimator architecture

The final estimator should not rely solely on fixed dataset values.

Preferred architecture:

```text
Dataset analysis
    -> prior estimate of R2
    -> prior estimate bandwidth
    -> circuit solver refinement
```

Recommended future solver variables:

```text
R2
X1
X2
```

Instead of:

```text
X1
X2 only
```

Use dataset-derived R2 as a soft prior:

```text
R2_prior = R3 * F_dataset(...)
```

Additional residual:

```text
r_R2 = log(R2 / R2_prior) / sigma_R2
```

This allows:

- Historical experience to guide the solution.
- The solver to adapt when a motor differs from the historical population.
- Better robustness for unusual or future designs.

This architecture should be considered the preferred end-state for the estimator.

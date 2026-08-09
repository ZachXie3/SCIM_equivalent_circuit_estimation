## Plan — SCIM Equivalent Circuit Estimation

### High-level goal

Estimate the single-cage per-phase equivalent-circuit parameters of a three-phase squirrel-cage induction motor from **typical nameplate / datasheet / test values** only:

- Rated voltage, frequency, poles, speed, HP, amps, power factor, efficiency
- No-load current and no-load power
- Locked-rotor current and locked-rotor torque
- Stator resistance at ambient temperature
- Connection type: Wye / Delta

The estimator shall use a **per-phase steady-state equivalent circuit** and solve only what is physically supported by the available data. Parameters that are not directly observable shall be handled with explicit engineering assumptions, iteration, or later tuning steps.

### Major direction change

The previous plan used a shunt-correction method to estimate total leakage reactance:

```text
Xtot = X1 + X2
```

That method is now **removed** from the plan.

Reason:

- The shunt-correction logic is not trustworthy.
- It uses terminal voltage and no-load current in a way that does not correctly represent the full-load internal rotor-path voltage/current relationship.
- It introduces hard-to-control bias into X1, X2, Xm, and downstream torque/current predictions.

The estimator shall no longer depend on:

- `Xtot_seed`
- `Xsum_seed_from_shunt`
- `Xsum_seed_realpart`
- Alpha split search
- `alpha = X1 / (X1 + X2)`

The concept of **alpha is discontinued**.

---

## Equivalent-circuit model basis

Use the per-phase stator-referred equivalent circuit:

```text
          R1_hot     jX1
Vph  ─────/\/\/──────j─────●──────────────
                            │
                            │
                         jXm│
                            │
                            ├──── R2/s + jX2 ────
                            │
                           N
```

At locked rotor, use the standstill rotor branch parameters:

```text
R3 = standstill rotor resistance
X3 = standstill rotor leakage reactance
XLR = X1 + X3
```

---

## Stage 1 — Completed (implemented)

Stage 1 (known quantities from input data) is **implemented** in `equivalent_circuit.py`:

- 1.1 Line-to-phase conversion for all operating points (Wye / Delta)
- 1.2 Synchronous speed and full-load slip (`n_s`, `omega_s`, `s_FL`)
- 1.3 Hot stator resistance `R1_hot` (copper temperature correction, K1 = 234.5)
- 1.4 Locked-rotor standstill rotor resistance `R3`
- 1.5 Locked-rotor reactance `XLR`
- 1.6 No-load reactance `X0` and magnetising reactance `Xm = X0 - X1`

The detailed formulas live in `spec/equivalent_circuit_estimation.md` (Part A). This section was removed from the plan once implemented.

---

## Stage 2 — Estimate R2 from R3 (dataset-driven empirical relationship)

### 2.1 Engineering intent

`R3` is rotor resistance at standstill. At standstill, rotor frequency equals rated supply frequency:

```text
f_r_start = f
```

At full load, rotor current frequency is slip frequency:

```text
f_r_fl = s_fl * f
```

Rotor resistance changes with rotor frequency due to skin effect. Therefore, use the locked-rotor rotor resistance `R3` and back-calculate the running rotor resistance `R2` at full-load slip frequency.

### 2.2 Skin-effect theory (why exact implementation is not feasible)

Use a frequency-dependent rotor AC resistance factor:

```text
R_ac(f_r) = R_dc * K_skin(f_r)
```

Therefore:

```text
R3 = R_dc * K_skin(f)
R2 = R_dc * K_skin(s_fl * f)
```

Eliminate `R_dc`:

```text
R2 = R3 * K_skin(s_fl * f) / K_skin(f)
```

The exact skin-effect relationship for a conductor uses the normal skin-effect equation:

```text
R_ac / R_dc = F(y)

y = h * sqrt(pi * mu * sigma * f_r)
```

For a rectangular bar approximation, one commonly used form is:

```text
F(y) = y * (sinh(2y) + sin(2y)) / (cosh(2y) - cos(2y))
```

Evaluating this formula requires:

```text
h        rotor bar depth         (geometry — not on the nameplate)
mu       bar permeability        (material — not available)
sigma    bar conductivity        (material — not available)
f_r      rotor frequency         (= f at standstill, = s_fl*f at full load)
```

The bar depth `h` is a design detail the user does not readily know, and `mu`/`sigma`
are material properties not published on nameplates or datasheets. Because these key
inputs are missing, the exact skin-effect equation **cannot be implemented** from the
available data. This is the reason R2 is instead derived from R3 via a dataset-driven
empirical relationship (§2.4), not from the analytic skin-effect formula.

### 2.3 Rationale for the dataset-driven empirical R2/R3 relationship

- The exact skin-effect model is infeasible: it needs bar depth and material properties
  (`h`, `mu`, `sigma`) that are not available to the user.
- The historical dataset (`examples_eq_results.csv`, see Stage 5) contains both `R2` (running) and `R3` (standstill) for
  fully-specified designs, so the `R2/R3` ratio can be modelled as an empirical function of
  readily-available nameplate quantities — slip, pole count, efficiency band.
- The skin-effect theory in §2.2 remains the physical justification for *why* `R2` and `R3`
  differ, but it is documented for reference only, not implemented.

Validation expectations (physical bounds, regardless of which model is used):

```text
K_skin(f) >= K_skin(s_fl*f) >= 1
R3 >= R2 > 0
```

If the calculated `R2` violates these expectations, flag the case and fall back to engineering review, not silent correction.

### 2.4 Dataset-driven R2/R3 model (R2 prior) — THE IMPLEMENTATION

The historical dataset (`examples_eq_results.csv`, see Stage 5) provides the empirical `R2/R3` relationship used to derive
R2 from R3; it seeds the solver or acts as a soft constraint.

Candidate models (power-law in log-log space), scored by MAE / RMSE / P10-P50-P90 of
predicted vs dataset R2:

```text
Model A:
R2/R3 = constant

Model B:
R2/R3 = f(s_fl)

Model C:
R2/R3 = f(s_fl, pole_count)

Model D:
R2/R3 = f(s_fl, pole_count, HorsePower)
```

For each model:

```text
Predict R2
Compare predicted R2 to dataset R2
Compute MAE, RMSE, and P10/P50/P90 error bands
```

Select the simplest model that captures most of the available accuracy.

Engineering expectation (validated against the dataset):

```text
Slip is likely the strongest predictor.
Pole count is likely the second strongest predictor.
Horsepower may provide smaller incremental improvement.
```

### 2.5 Decision (tentative) — adopt Model D

Analysis on the 4,180-row dataset (`typical_value/r2r3_report.md`, `§4`) supports Model D:

| Model | Predictors | RMSE (%) |
|---|---|---|
| A | constant | 71.8 |
| B | `s_fl` | 32.8 |
| C | `s_fl`, pole count | 36.4 |
| **D (selected)** | `s_fl`, pole count, HP | **26.0** |

Fitted power law (log-log fit, per pole count):

```
R2/R3 = exp(b) * s_fl^a * HP^c
R2 = R3 * exp(b) * s_fl^a * HP^c
```

Coefficients (see `typical_value/r2r3_report.md` §5):

| Poles | a | c | b |
|---|---|---|---|
| 2 | 0.744 | -0.050 | 2.464 |
| 4 | 0.634 | -0.129 | 2.277 |
| 6 | 0.466 | -0.154 | 1.629 |
| 8 | 0.452 | -0.123 | 1.393 |

**Status: tentative.** In-sample only; validate with a train/test split and against the
outcome of Stage 3 (X1, X2 solver) before finalising. R3 / R2 priors and bandwidths are
consumed by Stage 2.6.

Note: efficiency-band and enclosure splittings were dropped by review — enclosure has no
dependable nameplate proxy, and efficiency band is a messy metric. A grouping-constant
alternative (voltage / pole / HP bucket / slip bin tables) was scored and saved in the
report §6; it does not beat Model D.

Whatever model is chosen, it must respect the same physical bounds as §2.3:

```text
R3 >= R2 > 0
```

### 2.6 Long-term estimator architecture

The final estimator should not rely solely on fixed dataset values. Preferred architecture:

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

---

## Stage 3 — Solve X1 and X2 from full-load equations

### 3.1 Full-load phasor target

Use full-load phase current and full-load power factor:

```text
phi = arccos(PF_FL)
I_fl_vector = I_FL_ph * cos(phi) - j * I_FL_ph * sin(phi)
```

The current is lagging, so the reactive part is negative.

### 3.2 Equivalent-circuit full-load impedance

For a candidate pair `(X1, X2)`:

```text
Z_m = j * Xm
Z_2 = R2/s_fl + j * X2
Z_parallel = Z_m // Z_2
Z_fl_vector = R1_hot + j*X1 + Z_parallel
```

Where parallel operator means:

```text
A // B = A*B / (A + B)
```

Predicted full-load current:

```text
I_pred = V_ph / Z_fl_vector
```

Current-matching residuals can be formed from the complex current difference:

```text
res_I_real = real(I_pred - I_fl_vector)
res_I_imag = imag(I_pred - I_fl_vector)
```

### 3.3 Rotor current and torque equation

For the same candidate `(X1, X2)`, calculate rotor current using current division:

```text
I_2_vector = I_fl_vector / (j*Xm + R2/s_fl + j*X2) * (j*Xm)
```

Equivalent form using branch impedances:

```text
I_2_vector = I_fl_vector * Z_m / (Z_m + Z_2)
```

Full-load electromagnetic/mechanical output torque target:

```text
T_fl_calc = (3 * abs(I_2_vector)^2 * ((1 - s_fl)/s_fl) * R2) / w_s
```

Match this against the known full-load shaft torque:

```text
T_FL = P_out_hp * 745.7 / w_fl
```

### 3.4 Nonlinear solve target

Solve unknowns:

```text
X1, X2
```

Known values during each iteration:

```text
V_ph
I_FL_ph
PF_FL
R1_hot
R2
s_fl
Xm
T_FL
w_s
```

Use a nonlinear solver or bounded least-squares formulation. At minimum, enforce:

```text
X1 > 0
X2 > 0
Xm > X1
X3 = X_LR - X1 > 0
```

Recommended residual vector:

```text
r1 = real(I_pred - I_fl_vector) / I_FL_ph
r2 = imag(I_pred - I_fl_vector) / I_FL_ph
r3 = (T_fl_calc - T_FL) / T_FL
```

Because this is an overdetermined system for two unknowns, solve by least squares.

Optional additional residuals may be added later for efficiency and input power consistency, but do not reintroduce the removed shunt-correction Xtot seed.

---

## Stage 4 — Outer iteration for Xm, X3, X1, and X2 convergence

### 4.1 Iteration sequence

Use the following outer loop:

```text
1. Convert line quantities to phase quantities.
2. Calculate n_s, w_s, s_fl, T_FL.
3. Calculate R1_hot from R1_cold.
4. Calculate R3 from locked-rotor torque.
5. Calculate X_LR from locked-rotor current.
6. Calculate X_0 from no-load current and power.
7. Initialize X1 = 0.
8. Initialize Xm = X_0 - X1.
9. Initialize X3 = X_LR - X1.
10. Estimate R2 from R3 using the dataset-driven empirical relationship (§2.4).
11. Solve X1 and X2 from the full-load complex-current and torque equations.
12. Update Xm = X_0 - X1.
13. Update X3 = X_LR - X1.
14. Repeat steps 10-13 until X1 and Xm converge.
```

Note:

- `R2` may be recalculated each outer iteration if the dataset-driven model uses updated quantities.
- If the dataset-driven model depends only on slip and rated frequency, `R2` will remain unchanged during the outer loop.

### 4.2 Convergence criteria

Use both absolute and relative convergence checks:

```text
abs(X1_new - X1_old) < tol_abs
abs(Xm_new - Xm_old) < tol_abs
abs(X1_new - X1_old) / max(abs(X1_old), small) < tol_rel
abs(Xm_new - Xm_old) / max(abs(Xm_old), small) < tol_rel
```

Suggested defaults:

```text
tol_abs = 1e-6 ohm
tol_rel = 1e-5
max_outer_iter = 50
```

If convergence is not reached:

- Return the best available result.
- Set a convergence warning flag.
- Do not silently report the case as clean.

### 4.3 Required physical checks

After convergence, check:

```text
R1_hot > R1_cold > 0
R3 > R2 > 0
X1 > 0
X2 > 0
Xm > 0
Xm > X1
X_LR > X1
X3 = X_LR - X1 > 0
0 < s_fl < 1
```

Also calculate and report:

```text
Xtot = X1 + X2
XLR = X1 + X3
R2_R3_ratio = R2 / R3
Xm_X1_ratio = Xm / X1
```

These are diagnostic values only. They shall not be used as alpha-based constraints.

---

## Stage 5 — Compare estimated parameters against the ground-truth dataset

### 5.1 Objective

`typical_value/examples_eq_results.csv` (4,183 rows) carries two kinds of information:

1. **Nameplate / design inputs** the estimator is allowed to consume (voltage, frequency,
   poles, speed, HP, amps, no-load amps, connection, winding resistance at 105 °C), plus
2. **ground-truth per-phase circuit parameters** (`R1`, `R2`, `R3`, `X1`, `X2`, `X3`, `Xm`)
   computed by the same design-tool calculator (`examples_calculator.py`).

This stage runs the Stages 1-4 pipeline over every row, compares the predicted per-phase
circuit against the ground-truth columns, and reports where the estimator is close and where
it is systematically wrong.

### 5.2 Allowable inputs and what the dataset holds

The estimator may only consume the `MotorCase` fields listed in
`spec/equivalent_circuit_estimation.md` ("Input data"). Mapping:

| `MotorCase` field | name | dataset column | present? |
|---|---|---|---|
| `V_LL` | | `Voltage` | yes |
| `f` | Hz | `Frequency` | yes |
| `P` (poles) | | `PoleSpeed` | yes |
| `n_FL` | RPM | `RPM` | yes |
| `P_out` | HP | `HorsePower` | yes |
| `I_FL` | A | `Amps` | yes |
| `I_0` | A | `NoLoadAmps` | yes |
| `connection` | Wye/Delta | `Connection` | yes |
| `R1_cold` | Ω | reverse-corrected `WindingResistAt105` | derived (§5.2.1) |
| `PF_FL` | — | — | no → synthetic (§5.2.2) |
| `I_LR`, `T_LR`, `T_BD` | — | — | no → synthetic |
| `P_0`, `P_FW`, `P_core` | — | — | no → synthetic / default |
| `J` (inertia) | — | — | not needed by Stages 3-4 |

#### 5.2.1 Stator resistance from the 105 °C value

`WindingResistAt105` is `R1` at 105 °C. The estimator computes `R1_hot` from `R1_cold` using
`T_ambient_C + temp_rise_C` (defaults `25 °C` + `80 °C` = 105 °C). So feed an ambient value
that reproduces the 105 °C figure with those defaults:

```text
R1_cold = R1_dataset_105 ∘C * (T_ambient_C + K1_COPPER) / (T_ambient_C + temp_rise_C + K1_COPPER)
```

i.e. `R1_cold = R1_dataset_105 ∘C * (25 + 234.5) / (105 + 234.5)`. This keeps the estimator's
temperature path unchanged (defaults intact) and makes `R1_hot` match the dataset's `R1` column.

#### 5.2.2 Synthetic test inputs (forward simulation of ground truth)

The file has no `PF_FL`, `I_LR`, `T_LR`, `T_BD`, or no-load power. For a defensible
"recover known circuit" comparison, generate these as though they were measured on the
*known* motor: a perfect-instrument measurement of the ground-truth circuit:

```text
full-load :  s = s_FL(nameplate); Z_fl = R1_hot + jX1 + (jXm // (R2/s + jX2))
             I_FL stays the dataset Amps; PF_FL = cos(angle(Z_fl))
locked-rot:  I_LR   = V_ph / |R1_cold + R3 + j(X1 + X3)|
             T_LR   = [ 3 * I_LR² * R3 / ω_s ] / T_FL       (pu)
no-load  :  I_0    = |V_ph / (R1_hot + j(X1 + Xm))|
             P_0    = 3 · Re(V_ph · conj(I_0))
```

Then call `estimate_r2_from_r3` (Stage 2) and the Stage 3/4 solve as normal. Note that with
this construction the estimator's Stage-1 back-solves (`R3`, `XLR`, `X0`) become near-exact
by construction; the genuinely informative comparison is `R2` (dataset prior), `X1`, `X2`,
and the derived `Xm`, `X3`.

### 5.3 Comparison metrics

For each row run `EquivalentCircuitEstimator.fit()` and compute the relative error:

```text
err_pct(p) = 100 * (p_est - p_gt) / p_gt      for p in R2, R3, X1, X2, X3, Xm
```

Validated parameters are only those with a dataset counterpart: the file carries ground-truth
`R1, R2, R3, X1, X2, X3, Xm` columns. `R1` is excluded because it is a direct input to the
estimation (a simple, user-measurable resistance) rather than a recovered quantity; it is also
what the synthetic construction of §5.2.2 reproduces essentially exactly. `X0` and `XLR`
have no ground-truth column and are back-solves of the input data, so they are not part of
the error metric either.

Aggregate on converged rows only:

- MAE (%), RMSE (%), median, and P10 / P90 per parameter
- flag distribution (non_convergence, dataset_model_out_of_range, poor_full_load_* etc.)
  and the share of rows dropped
- cross-tables by pole count / voltage class / HP bucket / efficiency band / connection
- worst-outlier rows per parameter, for engineering review

Metrics are purely diagnostic; they only sanity-check %-error budgets.

### 5.4 Matrix-driven batch execution (no per-row Python loop)

With 4,183 rows the per-row estimator loop is slow. Plan a vectorised pipeline:

```text
Stage 1 + Stage 2 (closed form)   → numpy reduces over the whole table (no loop)
   V_ph, I_*_ph, n_s, ω_s, s_fl, R1_hot, T_FL, R3, X0, R2 prior
Stage 3 → (X1, X2) per row        → the most expensive non-linear solve; options
   1) keep scipy.optimize.least_squares per row and parallelize (multiprocessing)
   2) solve the 3-residual system with a vectorised Gauss-Newton fixed-point on
      (X1, X2) arrays (safe when starting each row at its Stage-2/Angle seed)
   3) warm-start X1, X2 from the typical leakage ratios per group (Stage 5) to cut the
      number of iterations
Stage 4 outer loop                → pure array updates (Xm = X0 - X1; X3 = XLR - X1);
                                     vectorised trivially
```

The validation harness must:

- implement Stage 1/2 and Stage 4 fully vectorised (pandas/numpy over columns);
- keep the per-row least-squares as the reference implementation and cross-check a
  subsample (e.g. 200 rows) matches the vectorised path within tolerance;
- report operator time for the vectorised run vs the per-row loop.

### 5.5 Deliverables

- `typical_value/validate.py` (batch runner + metric tables) with `pytest` sub-checks
- `typical_value/validation_report.md` generated by a report script
- report the per-parameter metrics, group breakdowns, flag/coverage counts, and runtime
  comparison
- any outlier families flagged for the estimator to rehandle
- recommended warning bands for the estimator, seeded from the observed spread of the
  parameter errors in §5.3 and consumed by Stage 2.6

Not started — this is the next implementation stage.

---

## Stage 6 — Later tuning extensions

Later, add optional tuning if additional performance data are available.

Potential future inputs:

- Locked-rotor torque, if not already used as primary input
- Breakdown torque `T_BD`
- Breakdown slip `s_BD`
- Additional measured load points
- Measured input power or efficiency at multiple loads

Future tuning may adjust:

```text
R2
X1
X2
Xm
dataset-model parameters (R2/R3 coefficients)
```

But the base estimator shall first work with the core process above.

Do not add these tuning steps until the base iterative estimator is validated.

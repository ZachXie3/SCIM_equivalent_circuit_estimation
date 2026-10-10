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

## Stage 2 — R2 from R3, and the circuit-solve question

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
- The historical dataset (`typical_value/data/eq_parameters.csv`, see Stage 5) contains both `R2` (running) and `R3` (standstill) for
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

### 2.4 R2/R3 estimation methods considered (the problem, tabled)

The running/standstill rotor-resistance ratio `R2/R3` is the input to the
Stages 3-4 solve: `R2 = ratio * R3` (§2.1). The exact skin-effect formula is
infeasible from nameplate data (§2.2), so the ratio has been modelled
empirically. Methods implemented so far:

| Method | Inputs | Form | Implementation | R2 test RMSE |
|---|---|---|---|---|
| A — constant | — | `R2/R3 = c` | `r2r3/models.py` | ~72% |
| B — slip power law | `s_fl` | `c * s^a` | `r2r3/models.py` | ~33% |
| C — per-pole slip law | `s_fl`, poles | per-pole `c * s^a` | `r2r3/models.py` | ~36% |
| D — per-pole slip+HP | `s_fl`, poles, HP | per-pole `c * s^a * HP^b` | `r2r3/models.py` | ~26% |
| slip+HP global law | `s_fl`, HP | `7.1 * s^0.57 / HP^0.12` | `r2r3/report.py` §11 | ~27% |
| slip-segment table | `s_fl` | constant per slip band | `r2r3/constant_bin.py` | ~32% |
| cluster — unweighted | 7 features | K-means lookup | `typical_value/cluster/` | ~32% |
| cluster — weighted | 7 features | NN-weighted K-means lookup | `typical_value/cluster/` | ~26% |

The three methods named for the current study are the **slip-segment table**,
the **slip+HP function**, and **clustering (unweighted / weighted)**. The best
achievable `R2/R3` prior error is ≈ 26% (relative RMSE on R2).

> Numbers are indicative: the power-law / table rows are in-sample from
> `reports/r2r3_report.md` (§4, §11); the cluster rows are held-out test from
> `reports/cluster_report.md`. The factor review that shaped the nameplate-only
> choices is in §2.7; the cluster analysis is documented in `cluster-plan.md`.

The `R2/R3` prior error is **one** of two independent error sources in the
estimator (see §2.5).

### 2.5 The fundamental question — is the circuit solve adequate?

The estimator's total error has two independent sources:

1. **`R2/R3` prior error** — feature-driven (the methods in §2.4; best ≈ 26%).
2. **Circuit-solve error** — given an accurate ratio, how well do the Stages 3-4
   equations recover `X1`, `X2`, `X3`, `Xm` (and `R2`/`R3` themselves) from the
   nameplate and test inputs?

**Question:** if we have an accurate `R2/R3`, with the current calculation
procedure in this plan, is it enough to calculate the equivalent circuit close
enough to ground truth?

**Experiment (the immediate next step):** decouple the two error sources by
feeding the estimator the **ground-truth `R2/R3` ratio for each record**,
bypassing the empirical prior:

- use only rows with `R2/R3 < 1` (a ratio ≥ 1 violates the physics bound
  `R2 <= R3` and is excluded);
- set `R2 = ratio_gt * R3` (equivalently, use the dataset `R2` column directly)
  and run the full Stages 1-4 pipeline;
- compare the estimated `X1`, `X2`, `X3`, `Xm`, `R3` against the dataset ground
  truth;
- aggregate relative errors (MAE / RMSE / P10-P90) per parameter and by subgroup
  (pole count, voltage class, HP bucket, connection).

Outcome decides where to focus next:

- small solve errors → the procedure is adequate; invest in the `R2/R3` prior.
- large solve errors → rework the Stage 3/4 solver first.

This experiment is run through the Stage 5 harness (§5).

#### 2.5.1 Findings from the first ground-truth run

First run of the experiment (ground-truth `R2`, real measurements, valid rows
only) — details in `reports/validation_report.md`:

- **`Xm` is recovered well** — ~3% RMSE (both connections).
- **`R2`** is recovered exactly where the physical clamp `R2 <= R3` does not
  bind (delta ~2% RMSE); wye is worse (~16%) because the real `R3` back-solve
  (from `T_LR = LockedRotorTorque/100`) occasionally undershoots and clamps R2.
- **`R3`** carries real measurement error (~12% median, RMSE ~31-52%) — the
  catalog `LockedRotorTorque` spec vs the design value, not a solver issue.
- **`X1`/`X2` are not identifiable** from the full-load terminal measurements:
  the solver pushes `X1` toward ~0 and lets `X2` absorb the total leakage
  (delta `X1` median −100%, `X2` +100%; wye `X1` +19%, `X2` +52%).

Diagnostic at the **true** `(X1, X2)`: the residuals are *not* zero —

```text
r1 (current, real)  ~ −0.8%     nameplate Amps vs the design circuit
r2 (current, imag)  ~ +5.8%     nameplate PowerFactor100 vs the design circuit
r3 (torque)         ~ +10%      same + solver-formula issues (below)
```

The nameplate `Amps` / `PowerFactor100` are catalog values that differ from the
exact design circuit by ~4-10%; because the `X1`/`X2` split is ill-conditioned,
this small inconsistency is amplified into large `X1`/`X2` errors.

**Solver-formula issues found (in addition to the identifiability):**

1. The torque residual computes the rotor current from the **measured** phasor
   `I2 = I_FL_vec * Zm/(Zm+Z2)` while the current residuals use the **predicted**
   phasor `I_pred = V/Z_fl` — two different circuit states in one objective.
2. The torque factor `(1−s)/s` computes `P_conv/ω_s` = shaft torque × `(1−s)`;
   it should be `1/s` (air-gap torque `3*I2²*R2/(s*ω_s)`).

A prototype with both fixes (same solver, ground-truth mode) improved recovery:
`X2` RMSE 3.36 → 1.06, `X1` median −0.68 → −0.15. See §5.6 for the full
correction plan.

#### 2.5.2 Conclusion

The current Stages 3-4 procedure is **adequate for `Xm`, `R2`, `R3`** but
**not for the `X1`/`X2` split**. Two distinct problems must be addressed before
the `R2/R3` prior becomes the limiting factor: (a) two solver-formula
inconsistencies (§5.6), and (b) the fundamental `X1`/`X2` identifiability,
which needs an additional constraint (leakage-split prior from the locked-rotor
data).

### 2.6 Long-term estimator architecture

The final estimator should not rely solely on fixed dataset values. Preferred architecture:

```text
Dataset analysis
    -> prior estimate of R2
    -> prior estimate bandwidth
    -> circuit solver refinement
```

**Prior candidates for the estimator** (production, when the true ratio is
unavailable): any of the §2.4 methods — Model D
(`R2/R3 = exp(b) * s_fl^a * HP^c` per pole; coefficients in
`reports/r2r3_report.md` §5), the global slip+HP law, or the weighted-cluster
lookup. The §2.5 experiment establishes the solver's accuracy when the prior is
exact — the ceiling any prior-based estimate can reach.

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

### 2.7 Factor review and the simple community-facing model (added after 2.4/2.5)

A factor review was run on the dataset (see `typical_value/reports/r2r3_report.md` §9-§11) to
answer: *can R2/R3 be treated as a constant per nameplate factor, so the estimator can
be simplified to a table a practising engineer can apply by hand?* Each factor was
scored against the Model D residual (per-pole `s_fl^a * HP^c`, baseline RMSE 24.1%).

| Factor | eta^2 (vs Model D resid) | RMSE with factor | verdict |
|---|---|---|---|
| Rotor bar geometry (`RotorSlotType`) | 0.249 | 20.5% | strong |
| Rotor material (`BarMaterial` = `ERMaterial`) | 0.145 | 21.5% | strong, 1:1 with slot shape |
| `NemaKVACode` | 0.085 | 22.4% | moderate; coarse bin of LR current |
| `NemaDesign` | 0.032 | 22.5% | weak |
| `Name` (efficiency level) | 0.033 | 23.5% | weak; material leakage |
| Pole count | 0.000 | — | already in per-pole coefficients |
| `FrameSize` | — | r = 0.07 | none |
| `LockedRotorAmps`/`Amps` | — | partial r = 0.16 | weak; pole proxy |

Conclusions:

- **Efficiency level ("Name"), NemaDesign, FrameSize: not usable** — spread within a
  group is far wider than the offset between groups.
- **NemaKVACode / LockedRotorAmps:Amps: not linear and mostly a pole proxy** — a
  coarse, redundant descriptor.
- **Pole count alone is not a constant** — a per-pole constant has RMSE 67-100%;
  slip must be in the formula.
- **Rotor bar geometry is the master variable** (`RotorSlotType`, internal
  diagnostic only — an end user does not know the slot shape), perfectly
  confounded with material (AL -> Dbl - Closed / Diamond / Oval; CU / Special ->
  Rectang / Trapeze). The five shapes behave very differently:

  | RotorSlotType | Material | n | geo-mean R2/R3 | slip exponent |
  |---|---|---|---:|---:|
  | Diamond | AL | 633 | 1.0000 (std 0) | 0.00 |
  | Oval | AL | 26 | 1.0000 (std 0) | 0.00 |
  | Trapeze | CU/Special | 265 | 0.226 | ≈0 |
  | Rectang | CU/Special | 148 | 0.320 | +0.16 |
  | Dbl - Closed | AL | 3120 | 0.281 | +0.82 |

  Diamond/Oval are shallow low bars with negligible skin effect (`R2 = R3` exactly in
  the design tool). Trapeze/Rectang are deep bars whose ratio is essentially a fixed
  geometry constant. Only Dbl - Closed (deep closed / double-cage) needs a slip law.
  This structure sets the achievable accuracy of any nameplate-only method but is
  not usable as a predictor itself.

**Suggested path — simple empirical method for the electrical community**

The recommended end-user estimator uses **only nameplate-available quantities**
(slip from RPM/poles/frequency, and HorsePower). No pole grouping, no rotor
material, no slot geometry (slot geometry explains the residual but is not known
to an end user; see `reports/r2r3_report.md` §10).

```text
R2/R3 = 7.1 * s_fl^0.57 / HorsePower^0.12        (s_fl = full-load slip)
```

- RMSE ≈ 27% (relative), P10–P90 ≈ −27% to +33% on R2 — within ~1 pp of the
  per-pole Model D (26.0%) with three coefficients and no grouping.
- If only slip is trusted: `R2/R3 = 20.3 * s_fl^0.92` (RMSE 32.8%).
- Clamp the result to `R3 >= R2` (the formula stays ≤ 1.0 over the whole dataset
  slip range; the estimator already clamps).
- If implemented in the estimator, replace `R2R3_MODEL_D` with this global law;
  Model D remains available as the historical reference.

**Approved alternative — slip-segment lookup table** (for hand use; no exponent
math). One constant `R2/R3` per slip band; `s_fl >= 5%` is taken as **1.0**
(trend extrapolation, only 12 rows above 5%):

| Slip band | n | R2/R3 | Bin RMSE (%) |
|---|---:|---:|---:|
| < 0.5% | 199 | 0.170 | 53.5 |
| 0.5–1% | 1526 | 0.202 | 28.7 |
| 1–2% | 1455 | 0.359 | 34.1 |
| 2–3% | 684 | 0.735 | 39.9 |
| 3–5% | 316 | 0.917 | 25.4 |
| ≥ 5% | 12 | 1.000 | 49.5 |

Overall RMSE ≈ 34%, MAE ≈ 26% (n = 4192). Roughly as accurate as the slip-only
power law; ~7 pp behind the continuous slip+HP formula. `reports/r2r3_report.md` §11.1
carries per-bin P10–P90 bands.

**Later addition — clustering methods** (`typical_value/cluster/`, see
`cluster-plan.md`): a K-means-style lookup over the seven nameplate features,
optionally with NN-learned per-feature weights and a feature-count penalty. The
unweighted version matches the slip-segment table (~32% held-out test RMSE);
the weighted version reaches ~26%, comparable to Model D, while still producing
a hand-applicable lookup table.

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
I_2_vector = I_pred / (j*Xm + R2/s_fl + j*X2) * (j*Xm)
```

Equivalent form using branch impedances:

```text
I_2_vector = I_pred * Z_m / (Z_m + Z_2)
```

Full-load electromagnetic/mechanical output torque target:

```text
T_fl_calc = (3 * abs(I_2_vector)^2 * (1/s_fl) * R2) / w_s
```

Match this against the known full-load shaft torque:

```text
T_FL = P_out_hp * 745.7 / w_fl
```

> **Correction note (§5.6):** the equations above are the *corrected* target —
> rotor current from the predicted phasor `I_pred` (one consistent circuit
> state for all residuals) and the air-gap torque factor `1/s` (not `(1−s)/s`).
> The current implementation in `equivalent_circuit.py` still uses the
> measured phasor `I_FL_vector` and `(1−s)/s`; it is pending the §5.6 solver
> correction.

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

`typical_value/data/eq_parameters.csv` (4,206 rows) carries two kinds of information:

1. **Nameplate / design inputs and test measurements** the estimator is allowed to
   consume (voltage, frequency, poles, speed, HP, amps, no-load amps, connection,
   winding resistance at 105 °C, power factor, locked-rotor current and torque,
   no-load power), plus
2. **ground-truth per-phase circuit parameters** (`R1`, `R2`, `R3`, `X1`, `X2`, `X3`, `Xm`)
   computed by the same design-tool calculator (`typical_value/eq_calculator.py`).

This stage runs the Stages 1-4 pipeline over every row, compares the predicted per-phase
circuit against the ground-truth columns, and reports where the estimator is close and where
it is systematically wrong.

**Row filter:** only rows with `R2/R3 < 1` are used. A ratio ≥ 1 (338 rows) violates the
physics bound `R2 <= R3` and is not valid data.

### 5.2 Allowable inputs and what the dataset holds

The estimator may only consume the `MotorCase` fields listed in
`spec/equivalent_circuit_estimation.md` ("Input data"). The dataset now carries
**all** of them as measured columns; only `R1_cold` and `T_LR` need conversion:

| `MotorCase` field | unit | dataset column | how supplied |
|---|---|---|---|
| `V_LL` | V | `Voltage` | as-is |
| `f` | Hz | `Frequency` | as-is |
| `P` (poles) | | `PoleSpeed` | as-is |
| `n_FL` | RPM | `RPM` | as-is |
| `P_out` | hp | `HorsePower` | as-is |
| `I_FL` | A | `Amps` | as-is |
| `I_0` | A | `NoLoadAmps` | as-is |
| `connection` | Wye/Delta | `Connection` | as-is |
| `R1_cold` | Ω | reverse-corrected `WindingResistAt105` | derived (§5.2.1) |
| `PF_FL` | — | `PowerFactor100` | as-is (fraction) |
| `I_LR` | A | `LockedRotorAmps` | as-is |
| `T_LR` | pu of `T_FL` | `LockedRotorTorque` | % of full-load torque → /100 |
| `P_0` | W | `NoLoadWatts` | as-is (3-phase watts) |

Fields **not** consumed by the algorithm, so not listed as inputs:

- `BreakDownTorque` (`T_BD`) — not used by Stages 1-4.
- `P_FW`, `P_core`, `J` (inertia) — not used by Stages 1-4.

#### 5.2.1 Stator resistance from the 105 °C value

`WindingResistAt105` **is** `R1` at 105 °C — the dataset gives the hot stator
phase resistance directly (per-phase after the `Jconn` conversion). In
production only `R1_cold` (ambient) is known. The estimator computes `R1_hot`
from `R1_cold` using `T_ambient_C + temp_rise_C` (defaults `25 °C` + `80 °C` =
105 °C), so feed the reverse-corrected ambient value:

```text
R1_cold = R1_dataset_105 ∘C * (T_ambient_C + K1_COPPER) / (T_ambient_C + temp_rise_C + K1_COPPER)
```

i.e. `R1_cold = R1_dataset_105 ∘C * (25 + 234.5) / (105 + 234.5)`. This keeps the estimator's
temperature path unchanged (defaults intact) and makes `R1_hot` match the dataset's `R1` column.

#### 5.2.2 Torque conversion and the test-measurement columns

No synthetic forward simulation is needed anymore — `PF_FL`, `I_LR`, `P_0`, and
`T_LR` are all measured columns (`PowerFactor100`, `LockedRotorAmps`,
`NoLoadWatts`, `LockedRotorTorque`). `LockedRotorTorque` is a **percentage of
full-load torque**, so:

```text
T_LR   = LockedRotorTorque / 100            (pu of T_FL)
```

(Equivalently: `T_FL = P_out * 745.7 / omega_fl` in N·m, then
`T_LR_abs = (LockedRotorTorque / 100) * T_FL`; the `T_FL` factor cancels in
the `R3` back-solve.) The `R3` back-solve is therefore a **real measurement**
(catalog spec vs design value, ~12% median), not a synthetic/exact value.

Then call the Stage 3/4 solve as normal. Two run modes:

- **Prior mode** (production view): `R2 = estimate_r2_from_r3(...)` — the
  empirical prior from §2.4; the `R2/R3` prior error is included.
- **Ground-truth mode** (the §2.5 experiment): `R2 = ratio_gt * R3` — bypass the
  prior entirely to isolate the circuit-solve error.

Per-phase convention (see `equivalent_circuit_estimation.md` §A1): the circuit
values are the wye-equivalent per-phase circuit (delta-to-wye `Z_y = Z_d/3`
via `Jconn` in `eq_calculator.py`), so `V_ph = V_LL/sqrt(3)` and
`I_ph = I_line` for **both** connections. Using `V_ph = V_LL` for delta (the
former convention) makes the delta circuit draw ~3x the nameplate current and
must not be used.

### 5.3 Comparison metrics

For each row run `EquivalentCircuitEstimator.fit()` and compute the relative error:

```text
err_pct(p) = 100 * (p_est - p_gt) / p_gt      for p in R2, R3, X1, X2, X3, Xm
```

Validated parameters are only those with a dataset counterpart: the file carries ground-truth
`R1, R2, R3, X1, X2, X3, Xm` columns. `R1` is excluded because it is a direct input to the
estimation (a simple, user-measurable resistance) rather than a recovered quantity. `X0` and
`XLR` have no ground-truth column and are back-solves of the input data, so they are not part
of the error metric either.

Run on the **valid subset only** (§5.1): rows with `R2/R3 < 1` and `slip > 0`.
Aggregate on converged rows only:

- MAE (%), RMSE (%), median, and P10 / P90 per parameter
- flag distribution (non_convergence, dataset_model_out_of_range, poor_full_load_* etc.)
  and the share of rows dropped
- cross-tables by pole count / voltage class / HP bucket / efficiency band / connection
- worst-outlier rows per parameter, for engineering review

Report the metrics for **both run modes** (§5.2.2):

- prior mode — the empirical `R2/R3` prior (production view; prior error included);
- ground-truth mode — `R2 = ratio_gt * R3` (§2.5 experiment; isolates the
  circuit-solve error). The ground-truth-mode errors are the headline answer to
  the §2.5 question.

Metrics are purely diagnostic; they only sanity-check %-error budgets.

### 5.4 Matrix-driven batch execution (no per-row Python loop)

With 4,206 rows the per-row estimator loop is slow. Plan a vectorised pipeline:

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
- `typical_value/reports/validation_report.md` generated by a report script
- report the per-parameter metrics, group breakdowns, flag/coverage counts, and runtime
  comparison
- any outlier families flagged for the estimator to rehandle
- recommended warning bands for the estimator, seeded from the observed spread of the
  parameter errors in §5.3 and consumed by Stage 2.6

Status: **implemented.** The harness (a) consumes the real measurement columns
(`PowerFactor100`, `LockedRotorAmps`, `NoLoadWatts`, `NoLoadAmps`,
`WindingResistAt105`, `LockedRotorTorque`) instead of synthetic inputs,
(b) applies the `R2/R3 < 1` filter, and (c) runs both `prior` and
`ground-truth` `R2/R3` modes. The §2.5 experiment is the ground-truth mode.
Findings are summarised in §2.5.1; the solver correction plan is §5.6.

Key points:

- `Xm` recovered ~3% RMSE; `R2` recovered except where the `R2 <= R3` clamp
  binds; `R3` carries real measurement error (`T_LR = LockedRotorTorque/100`).
- `X1`/`X2` are **not identifiable** from full-load terminal measurements alone
  — amplified by a small nameplate-vs-design inconsistency and two solver-formula
  issues (§5.6).
- **Per-phase convention resolved**: the circuit values are the wye-equivalent
  per-phase circuit (`Jconn = 3` for delta is correct), so `V_ph = V_LL/sqrt(3)`
  and `I_ph = I_line` for both connections. The former `V_ph = V_LL` for delta
  made the delta circuit imply ~3x the nameplate current — fixed in
  `equivalent_circuit.py` (§A1). Both connections are now internally consistent
  (`I_implied / I_nameplate` ≈ 0.96).

Open items carried forward: `BreakDownTorque` units (not used by the
algorithm); whether the residual `X1`/`X2` error is acceptable after §5.6.

### 5.6 Solver correction plan

The ground-truth experiment (§2.5.1) shows two solver-formula issues plus a
fundamental identifiability problem. Correct in this order:

1. **Consistent phasor state for the torque residual** — compute the rotor
   current from the **predicted** current `I_pred = V/Z_fl` (same circuit state
   as the current residuals), not from the measured `I_FL_vec`:
   `I2 = I_pred * Zm / (Zm + Z2)`. One consistent state for all three residuals.
2. **Torque factor** — use the air-gap torque `3*|I2|^2 * R2 / (s * omega_s)`
   (factor `1/s`), replacing `(1−s)/s` which computes `P_conv/omega_s` = shaft
   torque × `(1−s)`.
3. **Identifiability of the `X1`/`X2` split** — the total leakage is well
   determined but the split is not. Options (evaluate after 1-2):
   - add a **leakage-split soft prior** anchored by the locked-rotor data:
     `XLR = X1 + X3` is measured, and the design provides both `X2` and `X3`, so
     a prior on the rotor-leakage ratio `X2/X3` (skin-effect relation) or
     `X1/X2` separates the split;
   - **reparameterise** the solve as `(X1, Xtot)` or `(X1, X1/X2)` instead of
     `(X1, X2)`;
   - use **weighted / polar residuals** (`|I|`, angle) so catalog-`PF` noise
     (±~0.03) does not dominate the reactive residual.

Prototype result with fixes 1-2 (same solver, ground-truth mode): `X2` RMSE
3.36 → 1.06, `X1` median −0.68 → −0.15. Fixes 1-2 are the next implementation
step; fix 3 is the follow-up.

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

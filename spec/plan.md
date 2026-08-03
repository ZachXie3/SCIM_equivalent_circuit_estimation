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

## Stage 1 — Known quantities from input data

### 1.1 Line-to-phase conversion for all operating points

The per-phase equivalent circuit works with **phase** quantities. Nameplate/test data are normally line quantities.

Apply the same line-to-phase conversion consistently at all operating points:

- Full load
- Locked rotor
- No load

For Wye:

```text
V_ph = V_LL / sqrt(3)
I_ph = I_line
```

For Delta:

```text
V_ph = V_LL
I_ph = I_line / sqrt(3)
```

Required phase-current targets:

```text
I_FL_ph = line_to_phase_current(I_FL)
I_LR_ph = line_to_phase_current(I_LR)
I_0_ph  = line_to_phase_current(I_0)
```

All calculated currents from the equivalent circuit shall be compared against phase-current targets, not raw line currents.

---

### 1.2 Synchronous speed and full-load slip

Calculate:

```text
n_s  = 120 f / P
w_s  = 2*pi*n_s/60
s_fl = (n_s - n_FL) / n_s
```

Where:

- `n_s` = synchronous speed, RPM
- `w_s` = synchronous mechanical angular speed, rad/s
- `s_fl` = full-load slip

---

### 1.3 Stator resistance

Inputs provide the ambient stator phase resistance:

```text
R1_cold
```

Calculate hot stator resistance using the standard copper temperature correction:

```text
R1_hot = R1_cold * (T_ambient_C + temp_rise_C + K1) / (T_ambient_C + K1)
```

Use:

```text
K1 = 234.5
```

Default assumption unless otherwise specified:

```text
T_ambient_C = 25 deg C
temp_rise_C = 80 deg C
```

So the default hot resistance corresponds to approximately 105 deg C.

---

### 1.4 Locked-rotor standstill rotor resistance R3

Calculate standstill rotor resistance from locked-rotor torque and locked-rotor current:

```text
R3 = T_LR * w_s / (3 * I_LR_ph^2)
```

Where:

- `T_LR` must be in N.m if absolute torque is used.
- If input `T_LR` is per-unit of full-load torque, first calculate absolute locked-rotor torque:

```text
T_LR_abs = T_LR_pu * T_FL
R3 = T_LR_abs * w_s / (3 * I_LR_ph^2)
```

Full-load torque:

```text
w_fl = 2*pi*n_FL/60
T_FL = P_out_hp * 745.7 / w_fl
```

---

### 1.5 Locked-rotor reactance XLR

Calculate locked-rotor impedance from phase quantities:

```text
Z_LR = V_ph / I_LR_ph
R_LR = R1_cold + R3
X_LR = sqrt(Z_LR^2 - R_LR^2)
```

Use `R1_cold` because locked-rotor test data are normally taken near ambient temperature.

At this stage:

```text
X_LR = X1 + X3
```

But `X1` is not yet known. Therefore:

```text
X3 = X_LR - X1
```

During the initial pass, before X1 has been solved, use:

```text
X1 = 0
X3 = X_LR
```

After X1 is solved, update X3 and repeat the solution loop until convergence.

---

### 1.6 No-load reactance and magnetising reactance Xm

From no-load data:

```text
S_0 = 3 * V_ph * I_0_ph
Q_0 = sqrt(S_0^2 - P_0^2)
X_0 = 3 * V_ph^2 / Q_0
```

The no-load test gives total no-load reactance:

```text
X_0 = X1 + Xm
```

Therefore:

```text
Xm = X_0 - X1
```

Initial pass:

```text
X1 = 0
Xm = X_0
```

This is acceptable as the first estimate because normally:

```text
X1 << Xm
```

After solving X1 from the full-load equations, update:

```text
Xm = X_0 - X1
```

Then solve X1 and X2 again. Repeat until X1 and Xm converge.

---

## Stage 2 — Estimate R2 from R3 using rotor skin effect

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

---

### 2.2 Skin-effect relationship

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

This is the required implementation structure.

---

### 2.3 Reference skin-effect equation

Use the normal conductor skin-effect relationship as a reference model/comment in code:

```text
R_ac / R_dc = F(y)

y = h * sqrt(pi * mu * sigma * f_r)
```

For a rectangular bar approximation, one commonly used form is:

```text
F(y) = y * (sinh(2y) + sin(2y)) / (cosh(2y) - cos(2y))
```

Implementation note:

- The exact bar geometry and material properties may not be available from nameplate data.
- The first practical implementation may use a calibrated/default `K_skin(f_r)` model.
- The code shall isolate the skin-effect model in one function so it can be replaced later with a better rotor-bar model or empirical fit.

Required function shape:

```text
K_skin = skin_effect_factor(f_r, motor_case, optional_geometry_or_defaults)
R2 = R3 * skin_effect_factor(s_fl*f, case) / skin_effect_factor(f, case)
```

Validation expectations:

```text
K_skin(f) >= K_skin(s_fl*f) >= 1
R3 >= R2 > 0
```

If the calculated `R2` violates these expectations, flag the case and fall back to engineering review, not silent correction.

---

## Stage 3 — Solve X1 and X2 from full-load equations

### 3.1 Full-load phasor target

Use full-load phase current and full-load power factor:

```text
phi = arccos(PF_FL)
I_fl_vector = I_FL_ph * cos(phi) - j * I_FL_ph * sin(phi)
```

The current is lagging, so the reactive part is negative.

---

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

---

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

---

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
10. Estimate R2 from R3 using skin-effect back calculation.
11. Solve X1 and X2 from the full-load complex-current and torque equations.
12. Update Xm = X_0 - X1.
13. Update X3 = X_LR - X1.
14. Repeat steps 10-13 until X1 and Xm converge.
```

Note:

- `R2` may be recalculated each outer iteration if the skin-effect model uses updated quantities.
- If the skin-effect model depends only on slip and rated frequency, `R2` will remain unchanged during the outer loop.

---

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

---

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

## Stage 5 — Dataset and typical-value analysis

The historical dataset is still useful, but its role changes.

It shall no longer be used to define alpha or to split `Xtot`.

Instead, use the dataset to:

- Validate typical ranges of solved parameters.
- Check whether estimated `R2/R3` from the skin-effect model is realistic.
- Check typical `Xm/X1`, `X1/X2`, `X3/X2`, and `R2/R1` ranges as diagnostics.
- Identify outlier motor families where the solver needs special handling.
- Provide fallback engineering review bands, not primary formulas.

Required dataset checks:

```text
R1, R2, R3, X1, X2, X3, Xm > 0
R3 >= R2
Xm >> X1 for most cases
X3 = XLR - X1 > 0
```

Group statistics to review:

- Pole count
- Efficiency band
- Horsepower / frame size class
- Voltage class
- Connection type

Deliverable:

- A short analysis script
- Summary statistics of physical ratios
- Outlier list
- Recommended warning bands for the estimator

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
skin_effect_factor parameters
```

But the base estimator shall first work with the core process above.

Do not add these tuning steps until the base iterative estimator is validated.

---

## Implementation notes for coding agent

### Required removals from previous logic

Remove or stop using:

```text
alpha grid search
Xtot seed from shunt correction
X1 = alpha * Xtot
X2 = (1 - alpha) * Xtot
objective terms tied to the old shunt Xtot method
arbitrary R2 fallback = 0.05
hardcoded R2 bounds unrelated to motor scale
```

### Required new solver structure

Implement the estimator around this dependency chain:

```text
Input data
  -> line/phase conversion
  -> R1_hot, s_fl, T_FL
  -> R3 from locked-rotor torque
  -> X_LR from locked-rotor current
  -> X_0 from no-load data
  -> R2 from R3 using skin-effect back calculation
  -> solve X1, X2 from full-load current phasor and torque
  -> update Xm = X_0 - X1
  -> update X3 = X_LR - X1
  -> iterate to convergence
```

### Required output fields

Return at least:

```text
R1_cold
R1_hot
R2
R3
X1
X2
X3
Xm
X0
XLR
s_fl
T_FL
I_FL_ph
I_LR_ph
I_0_ph
converged
outer_iterations
solver_score
warning_flags
```

Diagnostic output:

```text
R2_R3_ratio
Xm_X1_ratio
X1_X2_ratio
X3_X2_ratio
```

### Warning flags

Include warning flags for:

```text
non_convergence
invalid_phase_conversion
invalid_locked_rotor_impedance
negative_or_zero_parameter
R2_greater_than_R3
Xm_not_greater_than_X1
X3_not_positive
poor_full_load_current_match
poor_full_load_torque_match
skin_effect_model_out_of_range
```

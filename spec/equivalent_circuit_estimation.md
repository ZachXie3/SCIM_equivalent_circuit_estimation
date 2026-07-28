# Spec 01 — SCIM Equivalent Circuit Estimation

## Overview

Estimate the single-cage equivalent-circuit parameters of a three-phase squirrel-cage induction motor from standard nameplate, no-load, and locked-rotor test data.

The estimator uses a **steady-state per-phase equivalent circuit** model and solves for the parameters via a hybrid approach:

1. Several parameters are computed **directly** from analytical formulas.
2. The remaining free parameter (rotor resistance `R2`) is solved by a 1-D bounded scalar optimisation for each candidate `alpha` split of total leakage reactance.
3. The best `alpha` is chosen by discrete grid search over a small set of candidate values.

---

## Input data (`MotorCase`)

| Field | Unit | Description |
|---|---|---|
| `V_LL` | V | Line-to-line rated voltage |
| `f` | Hz | Rated frequency |
| `P` | — | Number of poles |
| `n_FL` | RPM | Full-load speed |
| `P_out` | HP | Rated output power |
| `I_FL` | A | Full-load line current |
| `PF_FL` | — | Full-load power factor |
| `eta_FL` | — | Full-load efficiency (per unit) |
| `I_LR` | A | Locked-rotor (starting) line current |
| `T_LR` | pu | Locked-rotor torque (per unit of `T_FL`) |
| `T_BD` | pu | Breakdown torque (per unit of `T_FL`) |
| `s_BD` | — | Breakdown slip |
| `R_1` | Ω | Raw measured stator phase resistance at ambient |
| `I_0` | A | No-load line current |
| `P_0` | W | No-load three-phase power |
| `J` | kg·m² | Rotor inertia |
| `connection` | str | `'Y'` or `'D'` / `'DELTA'` (default `'Y'`) |
| `P_FW` | W | Friction & windage loss (optional) |
| `P_core` | W | Core loss (optional) |
| `T_ambient_C` | °C | Ambient temperature at test (default `25.0`) |
| `temp_rise_C` | °C | Estimated temperature rise at full load (default `80.0`) |

---

---

## ▸ Part A — Established formulas (no known issues)

All formulas assume the **per-phase** equivalent circuit below (stator & rotor referred to stator side).

```
     R1      X1          X2
   ──┬───────┬────┬───────┬────┐
     │       │    │       │    │
    ( ) Vph  │   Xm      R2/s │
     │       │    │       │    │
   ──┴───────┴────┴───────┴────┘
```

### A1. Per-phase voltage

```
        V_LL / √3    (Y connection)
V_ph =
        V_LL         (Δ/Delta connection)
```

### A2. Synchronous speed & full-load slip

```
n_s   = 120 f / P          [RPM]
ω_s   = 2π n_s / 60        [rad/s]
s_FL  = (n_s − n_FL) / n_s
```

### A3. Hot stator resistance

Copper temperature correction (coefficient `K1 = 234.5`):

```
R1_hot = R_1 · (T_ambient + T_rise + K1) / (T_ambient + K1)
```

### A4. Full-load torque

```
T_FL = P_out · 745.7 / ω_FL    [Nm]
```

where `ω_FL = 2π n_FL / 60`.

### A5. Locked-rotor reactance (`XLR`) — from locked-rotor impedance

Uses R3 from §B1 (accepted provisionally).

```
Z_LR = V_ph / I_LR
X_LR = √(Z_LR² − (R1_hot + R3)²)
```

`XLR` = `X1 + X3` (the `Xm` path is effectively open during LR).

---

## ▸ Part B — Known unknowns / pending engineering review

### B1. Locked-rotor resistance `R3` — from locked-rotor torque

```
R3 = T_LR · ω_s / (3 · I_LR²)
```

**Engineering caveats:**
- This formula assumes all `I_LR` flows through the rotor branch. Reasonable since `Xm >> Z_rotor` at standstill.
- However, `R3` and `R2` are **disconnected parameters** — there is no skin-effect or frequency-dependent model bridging standstill and running rotor resistance.
- Because `R3` and `XLR` are back-solved from LR data, the LR predictions (`ILR`, `TLR`) are **exact by construction**. LR data therefore provides *zero constraint* in the fit. The motor's starting behaviour is completely decoupled from its running behaviour.
- Needs resolution: unify `R3` and `R2` via a frequency-dependent rotor resistance model, or eliminate `R3` and solve for `R2` from LR constraints directly.

---

### B2. Magnetising reactance `Xm` — from no-load test

Ignore stator drop; use apparent power and reactive power:

```
S_0   = √3 · V_LL · I_0
Q_0   = √(S_0² − P_0²)
Xm    = 3 · V_ph² / Q_0
```

**Engineering caveats:**
- The quantity computed is the **total no-load reactance** `X_nl = X1_true + Xm_true`, not `Xm` alone. The formula `Xm = 3·V_ph²/Q_0` is equivalent to `X_nl = V_ph / (I_0·sin φ_0)`.
- In the no-load prediction (see §B5), `Z0 = R1_hot + j(X1_guess + self.Xm)` becomes `R1 + j(X1_guess + X_nl) = R1 + j(X1_guess + X1_true + Xm_true)`. This **double-counts X1**, underestimating `I0` and biasing the `P0` prediction.
- The full-load prediction also uses this inflated `Xm` value, redistributing current between the parallel branches (though the effect is smaller because `Xm >> X2`).
- Needs resolution: iterative separation — start with `Xm = X_nl`, solve alpha, update `Xm = X_nl - X1`, re-solve.

---

### B3. Total leakage reactance seed `Xtot = X1 + X2` — from full-load with shunt correction

Full-load current phasor:

```
φ_FL     = arccos(PF_FL)
I̲_FL    = I_FL · (cos φ_FL − j sin φ_FL)
```

Shunt current from no-load data:

```
I_w = P_0 / (3 · V_ph)
I_m = √(I_0² − I_w²)
I̲_sh ≈ I_w − j I_m
```

Effective series / rotor-path current:

```
I̲_sr = I̲_FL − I̲_sh
```

Branch impedance:

```
Z̲_sr = V_ph / I̲_sr
```

Seed:

```
Xtot_seed = Im(Z̲_sr)
```

**Engineering caveats:**
- **Z_sr is not the rotor-path impedance.** Rotor-path impedance requires the voltage at the parallel node, not the terminal voltage:
  ```
  Z_sr_correct = V_parallel / I_r
               = (V_ph − I_FL·(R1+jX1)) / (I_FL − I_mag_FL)
  ```
  The code uses `V_ph` and `I_0` (no-load phasor) instead.
- **Derivation of the systematic bias:**
  ```
  V_ph / (I_FL − I_0)  =  (R2/s + jX2) + (1 + I_mag/I_r)·(R1 + jX1)
  Im(...)               =  X1 + X2 + (I_mag/I_r)·X1      (10–25% high)
  ```
  The error term `(I_mag/I_r)·X1` is always positive — **this method always overestimates X1+X2**.
- **Phasor mismatch:** `I_0` is measured at no-load where voltage across the parallel branch ≈ `V_ph`. At full load, the voltage drops by `I_FL·(R1+jX1)`, so the real magnetising phasor differs in both magnitude and phase. The subtraction subtracts the wrong phasor.
- **Does not replace the objective function** — it's a seed that biases every downstream quantity. A better approach would be to make `Xtot` an optimisation variable alongside `alpha` and `R2`, or to iterate: solve → update Xtot from the now-known X1 → re-solve.

---

### B4. Full-load prediction (the `_predict` method)

The full-load equivalent-circuit calculation itself is standard:

```
Zm  = j Xm
Zr  = R2 / s_FL + j X2
Zpar = 1 / (1/Zm + 1/Zr)
Zin = R1_hot + j X1 + Zpar

IFL  = |V_ph / Zin|
PF   = Re(Zin) / |Zin|
Pin  = √3 · V_LL · IFL · PF
ETA  = P_out_hp · 745.7 / Pin
T_FL = 3 · |Vnode / Zr|² · (R2 / s_FL) / ω_s
```

**However** this correctness depends on the inputs `Xm` and `Xtot` being correct. Given the issues in §B2 and §B3, the full-load predictions (`IFL`, `PF`, `ETA`, `TFL`) inherit those biases.

---

### B5. No-load prediction

```
Z0  = R1_hot + j(X1 + Xm)
I0  = |V_ph / Z0|
P0  = 3 · I0² · R1_hot + P_core + P_FW
```

**Engineering caveat:** Inherits the `Xm` double-count from §B2. `I0` and `P0` predictions are systematically low.

---

### B6. Locked-rotor prediction

```
ILR = V_ph / √((R1_hot+R3)² + XLR²)
TLR = (3 / ω_s) · V_ph² · R3 / ((R1_hot+R3)² + XLR²)
```

**Engineering caveat:** Exact by construction — `R3` and `XLR` are back-solved to reproduce the measured `ILR` and `TLR`. These quantities provide no independent constraint.

---

### B7. Breakdown prediction

```
s_BDm = R2 / √(R1_hot² + Xtot²)
T_BD  = (3 / ω_s) · V_ph² · (R2 / s_BD) / ( (R1_hot + R2/s_BD)² + Xtot² )
```

**Engineering caveats:**
- Uses the simplified-circuit torque formula (neglects `Xm` path). This is a common approximation but its accuracy degrades for motors where `Xm` is not ≫ `Xtot`.
- Inherits any bias in `Xtot` from §B3.

---

### B8. Parameter fitting (`fit`)

**Tolerances** (reasonable — no known issue with these):

| Quantity | Tolerance |
|---|---|
| `I0` | `0.05 · I_0` |
| `P0` | `0.05 · P_0` |
| `TFL` | `0.05 · T_FL` |
| `IFL` | `0.03 · I_FL` |
| `PFFL` | `0.01 · PF_FL` |
| `ETAFL` | `0.01 · eta_FL` |

**Engineering caveats:**
- **Alpha grid** `{0.30, 0.35, 0.39, 0.40, 0.45, 0.50}` is coarse and uneven. The true `X1/Xtot` split may lie far from any candidate.
- **R2 bounds** `[1e-6, 0.2]` are hardcoded and motor-size-dependent. A small servo motor may have `R2` well below 1 mΩ; a large NEMA D motor may exceed 0.2 Ω.
- **Fallback** `R2 = 0.05` when SciPy is unavailable is arbitrary.
- **Objective mix:** the objective includes `I0`/`P0` which are biased by the Xm double-count. The minimiser may trade off errors across the six terms rather than converge to a physically meaningful solution.
- Known limitation from the earlier version not yet addressed: **Locked-rotor-based `XLR` bounds still pull the total leakage downward** because `X3 = XLR − X1` must be positive, which caps `X1 < XLR`. Combined with a biased `Xtot`, this can squeeze `X2`.

---

## Output

The fit returns a dictionary containing:
- `alpha_best`, `score`
- Optimised parameters: `R2`, `X1`, `X2`, `X3`, `Xm`, `R3`
- Predicted performance: `I0`, `P0`, `IFL`, `PF`, `ETA`, `TFL`, `ILR`, `TLR`, `sBD`, `TBD`
- Temperature-corrected stator resistance: `R1_hot`, `R1_raw`
- Xtot seed metadata: `X1_plus_X2`, `Xsum_seed_from_shunt`, `Xsum_seed_realpart`
- Complex seed phasors: `I_fl_complex`, `I_sh_complex`, `I_sr_complex`

---

## Summary of known unknowns (action items)

| # | Issue | Effect | Suggested approach |
|---|---|---|---|
| B1 | `R3`/`R2` dual resistance, LR predictions exact by construction | LR data provides zero constraint | Frequency-dependent rotor resistance model, or eliminate R3 |
| B2 | `Xm` = `X_nl` (includes X1), no-load prediction double-counts X1 | I0, P0 underestimated, optimizer biased | Iterative: Xm = X_nl − X1, re-solve |
| B3 | `Xtot` from shunt correction overestimates X1+X2 by 10–25% | All downstream quantities biased | Make Xtot an optimisation variable, or iterate |
| B7 | Breakdown formula neglects Xm | Minor, but compounds with Xtot bias | Use exact circuit formula |
| B8 | Coarse alpha grid, hardcoded R2 bounds, arbitrary fallback | May miss true optimum | Finer grid or continuous alpha optimisation; motor-adaptive R2 bounds |

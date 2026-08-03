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
| `R1_cold` | Ω | Raw measured stator phase resistance at ambient |
| `I_0` | A | No-load line current |
| `P_0` | W | No-load three-phase power |
| `J` | kg·m² | Rotor inertia |
| `connection` | str | `'Y'` / `'WYE'` or `'D'` / `'DELTA'` / `'Δ'` (case-insensitive, default `'Y'`) |
| `P_FW` | W | Friction & windage loss (not optional; assumed `0` if not available) |
| `P_core` | W | Core loss (not optional; estimated from no-load data if not available) |
| `T_ambient_C` | °C | Ambient temperature at test (default `25.0`) |
| `temp_rise_C` | °C | Estimated temperature rise at full load (default `80.0`) |

---

## Naming conventions & mapping to the example-design dataset

### Phase vs line quantities

All circuit parameters (`R`, `X`, `L`) and internal phasor quantities are **per-phase** values
referred to the stator. Nameplate/test data are **line** quantities:

| Quantity | Y / Wye | Δ / Delta |
|---|---|---|
| `V_ph` | `V_LL / √3` | `V_LL` |
| `I_ph` (from line current) | `I_L` | `I_L / √3` |
| 3-phase apparent power | `√3 · V_LL · I_L` | `√3 · V_LL · I_L` |

This applies at **every** operating point (full-load, locked-rotor, no-load) — see Task 1.4.

### Canonical circuit symbols

| Symbol | Meaning | Unit |
|---|---|---|
| `R1` | Stator phase resistance (hot) | Ω |
| `R1_cold` | Stator phase resistance at ambient (measured) | Ω |
| `R1_hot` | Temperature-corrected `R1` | Ω |
| `R2` | Running rotor resistance (referred) | Ω |
| `R3` | Standstill (locked-rotor) rotor resistance | Ω |
| `X1` | Stator leakage reactance | Ω |
| `X2` | Rotor leakage reactance (referred) | Ω |
| `X3` | Standstill (locked-rotor) rotor leakage reactance | Ω |
| `Xm` | Magnetising reactance | Ω |
| `XLR` | Locked-rotor reactance, `XLR = X1 + X3` | Ω |
| `Xtot` | Total leakage reactance, `Xtot = X1 + X2` | Ω |
| `alpha` | Split ratio `X1 / (X1 + X2)` | — |
| `Zbase` | Base impedance `V_LL / (√3 · I_FL)` | Ω |

### Mapping to the example-design results (`typical_value/examples_eq_results.csv`)

Typical values are extracted from **`examples_eq_results.csv`** (input columns + computed
per-phase circuit columns), not from the raw inputs. Each canonical symbol maps to a column of
that results file:

| Canonical | results column | Notes |
|---|---|---|
| `V_LL` | `Voltage` | Line-to-line voltage |
| `f` | `Frequency` | Hz |
| `P` (poles) | `PoleSpeed` | Column holds pole count, not speed |
| `connection` | `Connection` | Dataset uses `Wye` / `Delta` |
| `I_FL` | `Amps` | Full-load line current |
| `I_0` | `NoLoadAmps` | No-load line current |
| `Zbase` | `Zbase` | `V_LL / (√3 · I_FL)` |
| `R1` | `R1` | Hot stator phase resistance |
| `R2` | `R2` | Running rotor resistance (referred) |
| `R3` | `R3` | Standstill rotor resistance |
| `X1` | `X1` | Stator leakage reactance |
| `X2` | `X2` | Rotor leakage reactance (referred) |
| `X3` | `X3` | Standstill rotor leakage reactance |
| `Xm` | `Xm` | Magnetising reactance |
| `Xtot` | `X1 + X2` | Derived |
| `alpha` | `X1 / (X1 + X2)` | Derived split ratio |
| `XLR` | `X1 + X3` | Derived locked-rotor reactance |
| `XdPrime` | `XdPrime` | Transient reactance |
| `XdDoublePrime` | `XdDoublePrime` | Sub-transient reactance |
| `ShortCircuitTimeConstant` | `ShortCircuitTimeConstant` | s |
| `SubTransientTimeConstant` | `SubTransientTimeConstant` | s |
| `OpenCircuitTimeConstant` | `OpenCircuitTimeConstant` | s |
| `L1`, `L2`, `L3` | `L1`, `L2`, `L3` | H |
| `FirstCycleInrush` | `FirstCycleInrush` | pu |

The calculator derives `R1`, `R2`, `R3`, `X1`, `X2`, `X3` from the design-tool slot / bar /
end-ring inputs (retained in the results file under their original column names) using the
`Jconn` / `Xcorr` conventions of `EquivalentCircuitParameters Rev.5.xlsx`:

- `Jconn = 3` for Δ/Delta, `1` otherwise; `Xcorr = 2/3` for `PoleSpeed ≤ 4`, `1/2` otherwise
  (see `examples_calculator.py`).

`R1_hot` in `equivalent_circuit.py` uses default `T_ambient_C = 25` and `temp_rise_C = 80`, i.e.
105 °C — the same temperature as the dataset's `R1`.

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
        V_LL / √3    (Y / Wye connection)
V_ph =
        V_LL         (Δ/Delta connection)
```

The rated line voltage `V_LL` is the same at every operating point (full-load, locked-rotor,
no-load), so a single `V_ph` applies throughout.

### A1b. Per-phase currents

Nameplate/test currents are **line** values. The per-phase equivalent circuit works with
**phase** currents, converted once up front:

```
        I_line              (Y / Wye connection)
I_ph =
        I_line / √3         (Δ/Delta connection)
```

Applied to `I_FL`, `I_LR`, and `I_0`:

```
I_FL_ph = line_to_phase(I_FL)
I_LR_ph = line_to_phase(I_LR)
I_0_ph  = line_to_phase(I_0)
```

All predictions (`I0`, `IFL`, `ILR`) and fits are then compared against these per-phase targets.

### A2. Synchronous speed & full-load slip

```
n_s   = 120 f / P          [RPM]
ω_s   = 2π n_s / 60        [rad/s]
s_FL  = (n_s − n_FL) / n_s
```

### A3. Hot stator resistance

Copper temperature correction (coefficient `K1 = 234.5`):

```
R1_hot = R1_cold · (T_ambient + T_rise + K1) / (T_ambient + K1)
```

### A4. Full-load torque

```
T_FL = P_out · 745.7 / ω_FL    [Nm]
```

where `ω_FL = 2π n_FL / 60`.

### A5. Locked-rotor resistance `R3` — from locked-rotor torque

Uses the per-phase locked-rotor current `I_LR_ph`:

```
R3 = T_LR · ω_s / (3 · I_LR_ph²)
```

Notes:
- Assumes all `I_LR_ph` flows through the rotor branch — valid since `Xm >> Z_rotor` at standstill.
- Because `R3` and `XLR` are back-solved from LR data, the LR predictions (`ILR`, `TLR`) are
  **exact by construction** — LR data provides *zero constraint* on `R2`.
- **Skin-effect bridge (`R2` ↔ `R3`, TBD):** `R2` (running) and `R3` (standstill) are the same
  rotor resistance seen at different frequencies — rotor slip frequency vs standstill line
  frequency `f`:
  ```
  R2  → (skin effect at frequency f) → R3
  ```
  Use this to **verify** an estimated `R2` against `R3`, or **derive `R2` from `R3`** if `R2` is
  hard to obtain. **TBD:** model the frequency-dependent rotor-resistance factor.

### A6. Locked-rotor reactance (`XLR`) — from locked-rotor impedance

Calculated in sequence; uses `R3` from §A5 and the per-phase locked-rotor current `I_LR_ph`.
The stator resistance term is `R1_cold` at **ambient** (the locked-rotor test is run at ambient,
not hot):

```
Z_LR  = V_ph / I_LR_ph       1. locked-rotor impedance
R_LR  = R1_cold + R3         2. locked-rotor resistance (ambient stator + standstill rotor)
X_LR  = √(Z_LR² − R_LR²)     3. locked-rotor reactance
```

`XLR` = `X1 + X3` (the `Xm` path is effectively open during LR).

---

## ▸ Part B — Known unknowns / pending engineering review

### B1. Magnetising reactance `Xm` — from no-load test

Ignore stator drop; use apparent power and reactive power (from per-phase quantities,
`S_0 = 3 · V_ph · I_0_ph ≡ √3 · V_LL · I_0` for both connections). The no-load test yields the
**total no-load reactance** `X0 = X1_true + Xm_true`, not `Xm` alone:

```
S_0   = 3 · V_ph · I_0_ph
Q_0   = √(S_0² − P_0²)
X_0   = 3 · V_ph² / Q_0
```

The magnetising reactance follows once the `X1` split is known (later step):

```
X1  = alpha · Xtot        (Xtot from §B2, alpha from the fit)
Xm  = X_0 − X1
```

**Engineering caveats:**
- `X_0` is the **total no-load reactance** `X0 = X1_true + Xm_true`, not `Xm` alone. The formula
  `X0 = 3·V_ph²/Q_0` is equivalent to `X0 = V_ph / (I_0_ph·sin φ_0)`.
- **Double-count resolved:** the no-load prediction `Z0 = R1_hot + j(X1 + Xm)` now uses
  `Xm = X0 − X1`, so `Z0 = R1_hot + j X0` — X1 is no longer double-counted (previously the
  prediction was `R1 + j(X1 + X0)`, under-estimating `I0` and `P0`).
- `Xm` depends on `alpha` (via `X1 = alpha · Xtot`), so it is resolved inside `_predict` for each
  candidate `alpha`, not fixed up front.

---

### B2. Total leakage reactance seed `Xtot = X1 + X2` — from full-load with shunt correction

All phasors use **per-phase** currents (`I_FL_ph`, `I_0_ph`).

Full-load current phasor:

```
φ_FL     = arccos(PF_FL)
I̲_FL    = I_FL_ph · (cos φ_FL − j sin φ_FL)
```

Shunt current from no-load data:

```
I_w = P_0 / (3 · V_ph)
I_m = √(I_0_ph² − I_w²)
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

### B3. Full-load prediction (the `_predict` method)

The full-load equivalent-circuit calculation itself is standard:

```
Zm  = j Xm
Zr  = R2 / s_FL + j X2
Zpar = 1 / (1/Zm + 1/Zr)
Zin = R1_hot + j X1 + Zpar

IFL  = |V_ph / Zin|
PF   = Re(Zin) / |Zin|
Pin  = 3 · V_ph · IFL · PF          (per-phase basis; ≡ √3·V_LL·I_line·PF)
ETA  = P_out_hp · 745.7 / Pin
T_FL = 3 · |Vnode / Zr|² · (R2 / s_FL) / ω_s
```

**However** this correctness depends on the inputs `Xm` and `Xtot` being correct. Given the issues in §B1 and §B2, the full-load predictions (`IFL`, `PF`, `ETA`, `TFL`) inherit those biases.

---

### B4. No-load prediction

All currents per-phase; `I0` is the phase no-load current, compared against `I_0_ph`. Uses
`Xm = X0 − X1` (see §B1), so `Z0` simplifies to `R1_hot + j X0` (no X1 double-count).

```
Z0  = R1_hot + j(X1 + Xm) = R1_hot + j X0
I0  = |V_ph / Z0|
P0  = 3 · I0² · R1_hot + P_core + P_FW
```

**Engineering caveat:** `X0` is computed ignoring the stator drop, so `I0`/`P0` remain approximate.

---

### B5. Locked-rotor prediction

All currents per-phase; `ILR` is the phase locked-rotor current, compared against `I_LR_ph`.

```
ILR = V_ph / √((R1_hot+R3)² + XLR²)
TLR = (3 / ω_s) · V_ph² · R3 / ((R1_hot+R3)² + XLR²)
```

**Engineering caveat:** Exact by construction — `R3` and `XLR` are back-solved to reproduce the measured `ILR` and `TLR`. These quantities provide no independent constraint.

---

### B6. Breakdown-slip prediction

The nameplate breakdown slip `s_BD` is not used (removed from inputs). Only the model's own
breakdown-slip estimate is reported:

```
s_BDm = R2 / √(R1_hot² + Xtot²)
```

**Engineering caveats:**
- Uses the simplified-circuit formula (neglects `Xm` path). This is a common approximation but its accuracy degrades for motors where `Xm` is not ≫ `Xtot`.
- Inherits any bias in `Xtot` from §B2.
- No breakdown-torque prediction is currently made (the `T_BD` torque at a measured `s_BD` was dropped together with the `s_BD` input).

---

### B7. Parameter fitting (`fit`)

**Tolerances** (reasonable — no known issue with these; `I0`/`IFL` tolerances are against the per-phase targets `I_0_ph`/`I_FL_ph`):

| Quantity | Tolerance |
|---|---|
| `I0` | `0.05 · I_0_ph` |
| `P0` | `0.05 · P_0` |
| `TFL` | `0.05 · T_FL` |
| `IFL` | `0.03 · I_FL_ph` |
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
- Predicted performance: `I0`, `P0`, `IFL`, `PF`, `ETA`, `TFL`, `ILR`, `TLR`, `sBD`
- Stator resistance: `R1_hot` (temperature-corrected), `R1_cold` (ambient)
- Xtot seed metadata: `X1_plus_X2`, `Xsum_seed_from_shunt`, `Xsum_seed_realpart`
- Complex seed phasors: `I_fl_complex`, `I_sh_complex`, `I_sr_complex`

---

## Summary of known unknowns (action items)

| # | Issue | Effect | Suggested approach |
|---|---|---|---|
| B1 | `Xm` = `X0` (includes X1) until the `X1` split is known | `Xm` resolved per-`alpha` in `_predict`; fixed | `Xm = X0 − X1`, `X1 = alpha·Xtot` (done) |
| B2 | `Xtot` from shunt correction overestimates X1+X2 by 10–25% | All downstream quantities biased | Make Xtot an optimisation variable, or iterate |
| B6 | Breakdown-slip prediction `s_BDm` neglects Xm (torque prediction dropped with `s_BD` input) | Minor, but compounds with Xtot bias | Use exact circuit formula |
| B7 | Coarse alpha grid, hardcoded R2 bounds, arbitrary fallback | May miss true optimum | Finer grid or continuous alpha optimisation; motor-adaptive R2 bounds |

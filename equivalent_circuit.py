from __future__ import annotations

import cmath
import math
from dataclasses import dataclass

import numpy as np

try:
    from scipy.optimize import least_squares
    _HAS_SCIPY = True
except Exception:  # pragma: no cover - scipy expected in the target env
    least_squares = None
    _HAS_SCIPY = False

EPS = 1e-12
K1_COPPER = 234.5
HP_TO_WATT = 745.6998715822701

# Outer-iteration defaults (plan.md §4.2)
TOL_ABS = 1e-6   # ohm
TOL_REL = 1e-5
MAX_OUTER_ITER = 50


@dataclass
class MotorCase:
    V_LL: float
    f: float
    P: int
    n_FL: float
    P_out: float  # HP
    I_FL: float
    PF_FL: float
    eta_FL: float
    I_LR: float
    T_LR: float   # pu of T_FL
    T_BD: float   # pu of T_FL
    R1_cold: float
    I_0: float
    P_0: float
    J: float
    connection: str = 'Y'
    P_FW: float | None = None
    P_core: float | None = None
    T_ambient_C: float = 25.0
    temp_rise_C: float = 80.0


# ---------------------------------------------------------------------------
# Standalone R2-from-R3 model (plan.md Stage 2.4 / 2.5)
#
# Deliberately isolated so the empirical rotor-resistance relationship can be
# replaced in the future without touching the solver: update R2R3_MODEL_D
# and/or R2R3_MODEL_FALLBACK and the call sites keep working.
# ---------------------------------------------------------------------------

R2R3_MODEL_D = {
    # pole_count: {"a": slip exponent, "c": HP exponent, "b": intercept}
    2: {"a": 0.744, "c": -0.050, "b": 2.464},
    4: {"a": 0.634, "c": -0.129, "b": 2.277},
    6: {"a": 0.466, "c": -0.154, "b": 1.629},
    8: {"a": 0.452, "c": -0.123, "b": 1.393},
}

# Global Model B fallback for unseen pole counts (HP term dropped, c = 0).
R2R3_MODEL_FALLBACK = {"a": 0.9239, "b": 3.0090}

# Fitted envelope of the 4180-row training dataset (r2r3_report.md §3/§5).
DATASET_SLIP_RANGE = (0.0014, 0.1834)
DATASET_HP_RANGE = (0.4, 4600.0)


def estimate_r2_from_r3(
    R3: float,
    s_fl: float,
    pole_count: int,
    horsepower: float,
    model: dict[int, dict] | None = None,
    fallback: dict[str, float] | None = None,
) -> dict:
    """Estimate running rotor resistance R2 from standstill R3 (plan §2).

    Model power law, per pole count::

        R2/R3 = exp(b) * s_fl^a * HP^c

    Unseen pole counts fall back to the global model B (c = 0). The result
    respects the physical bound ``0 < R2 <= R3`` (plan §2.3); a violation is
    clamped and reported via ``clamped`` so the caller can flag it.

    Returns a dict with keys ``r2``, ``ratio``, ``pole_match``,
    ``out_of_range``, ``clamped``.
    """
    mdl = model if model is not None else R2R3_MODEL_D
    fb = fallback if fallback is not None else R2R3_MODEL_FALLBACK

    coef = mdl.get(int(pole_count))
    pole_match = coef is not None
    a = coef["a"] if pole_match else fb["a"]
    b = coef["b"] if pole_match else fb["b"]
    c = coef["c"] if pole_match else 0.0

    out_of_range = not (
        DATASET_SLIP_RANGE[0] <= s_fl <= DATASET_SLIP_RANGE[1]
        and DATASET_HP_RANGE[0] <= horsepower <= DATASET_HP_RANGE[1]
    )

    if min(s_fl, horsepower) <= 0.0:
        ratio = 0.0
    else:
        ratio = math.exp(b + a * math.log(s_fl) + c * math.log(horsepower))

    r2_raw = R3 * ratio
    clamped = bool(R3 > 0.0 and r2_raw >= R3)
    r2 = min(max(r2_raw, 0.0), R3) if R3 > 0.0 else 0.0
    return {
        "r2": r2,
        "ratio": float(ratio),
        "pole_match": bool(pole_match),
        "out_of_range": bool(out_of_range),
        "clamped": bool(clamped),
    }


class EquivalentCircuitEstimator:
    """Steady-state-oriented estimator (plan.md Stages 3 and 4).

    Pipeline (plan.md §4.1):
      1. line/phase conversion                        (per-phase V and I)
      2. n_s, omega_s, s_FL, T_FL
      3. R1_hot from R1_cold                         (temperature corrected)
      4. R3 from locked-rotor torque                  (standstill rotor R)
      5. X_LR from locked-rotor current               (R1_cold + R3 path)
      6. X_0 from no-load current and power           (total no-load reactance)
      7-14. outer iteration:
         R2 = estimate_r2_from_r3(...)                (stage 2 dataset model)
         -> solve (X1, X2) from the full-load current
            phasor + torque equations (stage 3)
         -> Xm = X0 - X1, X3 = X_LR - X1
         until X1 and Xm converge (tol_abs / tol_rel / max_outer)

    The old alpha grid and the shunt-correction Xtot seed are removed per the
    implementation plan (plan.md "Required removals").
    """

    def __init__(self, case: MotorCase):
        self.case = case
        self.V_ph = self._per_phase_voltage()
        # Per-phase currents: I_ph = I_line (Y), I_ph = I_line/√3 (Δ).
        self.I_FL_ph = self._line_to_phase_current(case.I_FL)
        self.I_LR_ph = self._line_to_phase_current(case.I_LR)
        self.I_0_ph = self._line_to_phase_current(case.I_0)
        self.n_s = 120.0 * case.f / case.P
        self.omega_s = 2.0 * math.pi * self.n_s / 60.0
        self.s_FL = (self.n_s - case.n_FL) / self.n_s
        self.T_FL = case.P_out * HP_TO_WATT / max(2.0 * math.pi * case.n_FL / 60.0, EPS)
        self.T_LR_N = case.T_LR * self.T_FL
        self.T_BD_N = case.T_BD * self.T_FL

        self.R1_hot = self._correct_r1_to_hot()
        self.R3 = self._estimate_r3_from_locked_rotor()
        self._lr_impedance_invalid = False
        self.XLR = self._estimate_xlr_from_locked_rotor()
        self.X0 = self._estimate_x0_from_no_load()

        # Full-load current phasor (plan §3.1): lagging, imag negative.
        phi = math.acos(max(min(case.PF_FL, 1.0), -1.0))
        self.I_FL_vec = cmath.rect(self.I_FL_ph, -phi)

    @property
    def _is_delta(self) -> bool:
        c = self.case.connection.strip().upper()
        if c in {'Y', 'WYE'}:
            return False
        if c in {'D', 'DELTA', 'Δ'}:
            return True
        raise ValueError(f'Unsupported connection: {self.case.connection!r}')

    def _per_phase_voltage(self) -> float:
        # V_LL -> V_ph: V_LL/√3 (Y), V_LL (Δ).
        if self._is_delta:
            return self.case.V_LL
        return self.case.V_LL / math.sqrt(3.0)

    def _line_to_phase_current(self, i_line: float) -> float:
        return i_line / math.sqrt(3.0) if self._is_delta else i_line

    def _correct_r1_to_hot(self) -> float:
        ta = self.case.T_ambient_C
        tb = ta + self.case.temp_rise_C
        return self.case.R1_cold * (tb + K1_COPPER) / (ta + K1_COPPER)

    def _estimate_x0_from_no_load(self) -> float:
        # Total no-load reactance X0 = X1_true + Xm_true (not Xm alone).
        # S0 computed from per-phase quantities (3*V_ph*I_0_ph is identical
        # to √3*V_LL*I_0 for both connections).
        s0 = 3.0 * self.V_ph * self.I_0_ph
        q0 = math.sqrt(max(s0 ** 2 - self.case.P_0 ** 2, 0.0))
        return 3.0 * self.V_ph ** 2 / max(q0, EPS)

    def _estimate_r3_from_locked_rotor(self) -> float:
        # R3 (plan §4.1 step 4) is the standstill rotor resistance back-solved
        # from the locked-rotor torque relation. R3 and X_LR are back-solved
        # from LR data, so predicted ILR/TLR always match the measurements;
        # the dataset model bridges R2 <-> R3.
        return self.T_LR_N * self.omega_s / max(3.0 * self.I_LR_ph ** 2, EPS)

    def _estimate_xlr_from_locked_rotor(self) -> float:
        # X_LR (phase) from the locked-rotor test (plan §4.1 step 5).
        # The LR test is run cold, so R1_cold (not R1_hot) is used.
        r_lr = self.case.R1_cold + self.R3
        z_lr = self.V_ph / max(self.I_LR_ph, EPS)
        z_sq = z_lr ** 2 - r_lr ** 2
        if z_sq <= 0.0:
            self._lr_impedance_invalid = True
            return EPS
        self._lr_impedance_invalid = False
        return math.sqrt(z_sq)

    @staticmethod
    def _zpar(z1: complex, z2: complex) -> complex:
        return 1.0 / (1.0 / z1 + 1.0 / z2)

    # -- Stage 3: (X1, X2) interior solve ---------------------------------

    def _residuals_x1x2(self, x: np.ndarray, R2: float, Xm: float) -> np.ndarray:
        """Full-load residual vector [r1, r2, r3] for candidate (X1, X2).

        r1, r2: real/imag current-matching error (plan §3.1), normalised by I_FL_ph.
        r3:     full-load torque error (plan §3.3), normalised by T_FL.
        Xm is fixed within one interior solve and refreshed in the outer loop.
        """
        X1, X2 = float(x[0]), float(x[1])
        if Xm <= 0.0 or X1 >= 0.999 * min(Xm, self.XLR) or X2 <= 0.0:
            return np.array([1e12, 1e12, 1e12])

        Zm = complex(0.0, Xm)
        Z2 = complex(R2 / max(self.s_FL, EPS), X2)
        Zpar = self._zpar(Zm, Z2)
        Zfl = complex(self.R1_hot, X1) + Zpar
        I_pred = self.V_ph / Zfl

        r1 = (I_pred.real - self.I_FL_vec.real) / max(self.I_FL_ph, EPS)
        r2 = (I_pred.imag - self.I_FL_vec.imag) / max(self.I_FL_ph, EPS)

        # Rotor current by current division (plan §3.3), using the measured
        # load current phasor, then the full-load torque equation.
        I2 = self.I_FL_vec * Zm / max(abs(Zm + Z2), EPS)
        T_calc = (
            3.0 * abs(I2) ** 2 * ((1.0 - self.s_FL) / max(self.s_FL, EPS)) * R2
            / max(self.omega_s, EPS)
        )
        r3 = (T_calc - self.T_FL) / max(self.T_FL, EPS)
        return np.array([r1, r2, r3])

    def _solve_x1_x2(self, R2: float, Xm: float) -> dict | None:
        # Bounds (plan §3.4): X1>0, X2>0, Xm > X1, X3 = X_LR - X1 > 0.
        x1_hi = 0.999 * min(Xm, self.XLR)
        x2_hi = max(2.0 * self.X0, 1.0)
        lo = np.array([1e-9, 1e-9])
        hi = np.array([x1_hi, x2_hi])

        if not _HAS_SCIPY:
            raise RuntimeError("Stage 3 solver requires scipy.optimize.least_squares")

        best = None
        for x1_frac in (0.02, 0.05, 0.15):
            for x2_frac in (0.02, 0.05, 0.15):
                x0 = np.array([x1_frac * x1_hi, x2_frac * x2_hi])
                try:
                    res = least_squares(
                        self._residuals_x1x2, x0, args=(R2, Xm),
                        bounds=(lo, hi), method="trf",
                        xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=400,
                    )
                except Exception:
                    continue
                score = float(np.sum(res.fun ** 2))
                if best is None or score < best["score"]:
                    best = {
                        "X1": float(res.x[0]),
                        "X2": float(res.x[1]),
                        "score": score,
                    }
        return best

    # -- Stage 4: outer iteration ------------------------------------------

    def fit(self) -> dict:
        warnings = set()

        # Validate phase conversion early (plan warning flag list).
        if not (math.isfinite(self.I_FL_ph) and self.I_FL_ph > 0.0 and
                math.isfinite(self.V_ph) and self.V_ph > 0.0):
            warnings.add("invalid_phase_conversion")
        if getattr(self, "_lr_impedance_invalid", False):
            warnings.add("invalid_locked_rotor_impedance")

        # R2 prior from the standalone dataset model. It is constant during
        # the outer loop for Models B/D, but is recomputed here so the loop
        # stays generic if the model gains X1-dependent predictors.
        r2_info = estimate_r2_from_r3(
            self.R3, self.s_FL, self.case.P, self.case.P_out
        )
        if r2_info["out_of_range"]:
            warnings.add("dataset_model_out_of_range")
        if not r2_info["pole_match"]:
            warnings.add("dataset_model_fallback_used")
        R2 = r2_info["r2"]
        if r2_info["clamped"]:
            warnings.add("R2_greater_than_R3")
        if R2 <= 0.0 or self.R3 <= 0.0:
            warnings.add("negative_or_zero_parameter")

        # Initialisation (plan §4.1 steps 7-9).
        X1 = 0.0
        Xm = self.X0 - X1
        best = None

        for it in range(1, MAX_OUTER_ITER + 1):
            X1_old, Xm_old = X1, Xm
            sol = self._solve_x1_x2(R2, Xm)
            if sol is None:
                warnings.add("non_convergence")
                break

            X1 = sol["X1"]
            X2 = sol["X2"]
            score = sol["score"]
            Xm_new = self.X0 - X1
            X3_new = self.XLR - X1

            best = {
                "X1": X1,
                "X2": X2,
                "Xm": Xm_new,
                "X3": X3_new,
                "R2": R2,
                "score": score,
                "outer_iterations": it,
            }

            conv_abs = abs(X1 - X1_old) < TOL_ABS and abs(Xm_new - Xm_old) < TOL_ABS
            conv_rel = (
                abs(X1 - X1_old) / max(abs(X1_old), EPS) < TOL_REL
                and abs(Xm_new - Xm_old) / max(abs(Xm_old), EPS) < TOL_REL
            )
            Xm, X3 = Xm_new, X3_new
            if conv_abs and conv_rel:
                best["converged"] = True
                break

        if best is None:
            return {
                "converged": False,
                "outer_iterations": 0,
                "solver_score": float("inf"),
                "warning_flags": sorted(warnings),
            }

        if not best.get("converged", False):
            warnings.add("non_convergence")

        # --- Post-solve physical checks (plan §4.3) ------------------------
        X1, X2, Xm, X3 = best["X1"], best["X2"], best["Xm"], best["X3"]
        R1_cold = self.case.R1_cold

        if self.R1_hot <= R1_cold or R1_cold <= 0.0:
            warnings.add("invalid_temperature_correction")
        if self.R3 <= R2 or self.R3 <= 0.0:
            warnings.add("R3_not_greater_than_R2")
        if min(X1, X2, Xm, X3, R2) <= 0.0:
            warnings.add("negative_or_zero_parameter")
        if Xm <= X1:
            warnings.add("Xm_not_greater_than_X1")
        if X3 <= 0.0:
            warnings.add("X3_not_positive")
        if not (0.0 < self.s_FL < 1.0):
            warnings.add("slip_out_of_range")

        # Full-load match quality (recomputed residual vector).
        r_final = self._residuals_x1x2(np.array([X1, X2]), R2, Xm)
        current_err = math.hypot(r_final[0], r_final[1])
        torque_err = abs(r_final[2])
        if current_err > 0.03:
            warnings.add("poor_full_load_current_match")
        if torque_err > 0.05:
            warnings.add("poor_full_load_torque_match")

        # --- Assemble output (plan "Required output fields") ---------------
        Xtot = X1 + X2
        out = {
            "R1_cold": R1_cold,
            "R1_hot": self.R1_hot,
            "R2": R2,
            "R3": self.R3,
            "X1": X1,
            "X2": X2,
            "X3": X3,
            "Xm": Xm,
            "X0": self.X0,
            "XLR": self.XLR,
            "s_fl": self.s_FL,
            "T_FL": self.T_FL,
            "I_FL_ph": self.I_FL_ph,
            "I_LR_ph": self.I_LR_ph,
            "I_0_ph": self.I_0_ph,
            "converged": bool(best.get("converged", False)),
            "outer_iterations": int(best["outer_iterations"]),
            "solver_score": float(best["score"]),
            "warning_flags": sorted(warnings),
            "R2_R3_ratio": R2 / max(self.R3, EPS),
            "Xm_X1_ratio": Xm / max(X1, EPS),
            "X1_X2_ratio": X1 / max(X2, EPS),
            "X3_X2_ratio": X3 / max(X2, EPS),
            "Xtot": Xtot,
        }
        return out
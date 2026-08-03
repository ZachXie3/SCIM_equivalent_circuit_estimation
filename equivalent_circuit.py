from __future__ import annotations

import cmath
import math
from dataclasses import dataclass

try:
    from scipy.optimize import minimize_scalar
    _HAS_SCIPY = True
except Exception:
    minimize_scalar = None
    _HAS_SCIPY = False

EPS = 1e-12
K1_COPPER = 234.5
HP_TO_WATT = 745.6998715822701


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
    R1_cold: float    # raw measured stator phase resistance at ambient
    I_0: float
    P_0: float
    J: float
    connection: str = 'Y'
    P_FW: float | None = None
    P_core: float | None = None
    T_ambient_C: float = 25.0
    temp_rise_C: float = 80.0


class EquivalentCircuitEstimator:
    """Steady-state-oriented estimator.

    *** KNOWN UNKNOWNS — see per-method comments for details ***
    - R3 from LR torque: disconnected from R2 (dual rotor-resistance model).
      LR predictions (ILR, TLR) are exact by construction, providing zero constraint.
      A skin-effect bridge R2 <-> R3 is proposed (TBD).
    - Xtot from shunt correction: overestimates X1+X2 by (I_mag/I_r)·X1 (~10-25%).
    - Alpha grid: coarse, uneven (6 values). R2 bounds: hardcoded [1e-6, 0.2].

    Logic (may change once unknowns are resolved):
    - Fix X0 (total no-load reactance) from the no-load test.
    - Fix Xtot = X1 + X2 from the full-load-with-shunt-current formula.
    - Compute R3 from LR torque/current relation.
    - Use XLR from LR current relation (with R1_cold + R3).
    - Inside _predict: X1 = alpha·Xtot, Xm = X0 - X1 (no X1 double-count).
    - Iterate alpha only (outer scan), while solving R2 for each alpha.
    """

    def __init__(self, case: MotorCase):
        self.case = case
        self.V_ph = self._per_phase_voltage()
        # Convert nameplate/test line currents to per-phase currents so that the
        # per-phase circuit predictions are compared on a consistent basis:
        #   I_ph = I_line        (Y / Wye)
        #   I_ph = I_line / √3   (Δ / Delta)
        self.I_FL_ph = self._line_to_phase_current(case.I_FL)
        self.I_LR_ph = self._line_to_phase_current(case.I_LR)
        self.I_0_ph = self._line_to_phase_current(case.I_0)
        self.n_s = 120.0 * case.f / case.P
        self.omega_s = 2.0 * math.pi * self.n_s / 60.0
        self.s_FL = (self.n_s - case.n_FL) / self.n_s
        self.R1_hot = self._correct_r1_to_hot()
        self.T_FL = case.P_out * HP_TO_WATT / max(2.0 * math.pi * case.n_FL / 60.0, EPS)
        self.T_LR_Nm = case.T_LR * self.T_FL
        self.T_BD_Nm = case.T_BD * self.T_FL
        self.X0 = self._estimate_x0_from_no_load()
        self.R3 = self._estimate_r3_from_locked_rotor()
        self.XLR = self._estimate_xlr_from_locked_rotor()
        self.Xtot = self._estimate_xtot_from_full_load_with_shunt()

    @property
    def _is_delta(self) -> bool:
        c = self.case.connection.strip().upper()
        if c in {'Y', 'WYE'}:
            return False
        if c in {'D', 'DELTA', 'Δ'}:
            return True
        raise ValueError(f'Unsupported connection: {self.case.connection!r}')

    def _per_phase_voltage(self) -> float:
        # V_LL -> V_ph: V_LL/√3 (Y), V_LL (Δ). Applies at every operating point
        # (full-load, locked-rotor, no-load) since the rated voltage is the same.
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
        # Total no-load reactance X0 = X1_true + Xm_true. This is NOT the
        # magnetising reactance Xm alone. The true magnetising reactance is
        #   Xm = X0 - X1   (X1 = alpha * Xtot, fixed in a later step),
        # so the no-load impedance Z0 = R1 + j(X1 + Xm) simplifies to R1 + j X0
        # and X1 is NOT double-counted.
        # No-load per-phase reactance from 3-phase no-load data. S0 computed from
        # per-phase quantities (3·V_ph·I_0_ph ≡ √3·V_LL·I_0 for both connections),
        # so 3·V_ph²/Q0 yields X0 = V_ph/I_m_phase consistently for Y and Δ.
        s0 = 3.0 * self.V_ph * self.I_0_ph
        q0 = math.sqrt(max(s0**2 - self.case.P_0**2, 0.0))
        return 3.0 * self.V_ph**2 / max(q0, EPS)

    def _estimate_r3_from_locked_rotor(self) -> float:
        # KNOWN-UNKNOWN: R3 is the standstill rotor resistance derived from LR torque,
        # but it is disconnected from R2 (running rotor resistance). No skin-effect
        # model bridges the two. Because R3 and XLR are back-solved from LR data,
        # ILR and TLR predictions always exactly match measured values — they provide
        # zero constraint on the fit. A frequency-dependent rotor resistance model
        # would unify R3 and R2.
        return self.T_LR_Nm * self.omega_s / max(3.0 * self.I_LR_ph**2, EPS)

    def _estimate_xlr_from_locked_rotor(self) -> float:
        # R_LR = R1_cold + R3: the locked-rotor test is at ambient, so use
        # the raw stator resistance R1_cold, not the hot value.
        rlr = self.case.R1_cold + self.R3
        zlr = self.V_ph / max(self.I_LR_ph, EPS)
        return math.sqrt(max(zlr**2 - rlr**2, EPS))

    def _estimate_xtot_from_full_load_with_shunt(self) -> float:
        # KNOWN-UNKNOWN: This method systematically overestimates X1+X2.
        #
        # Derivation of the bias:
        #   Z_sr = V_ph / (I_FL - I_0)
        #        = (R2/s + jX2) + (1 + I_mag/I_r)*(R1 + jX1)
        #   Im(Z_sr) = X1 + X2 + (I_mag/I_r)*X1   (10-25% high)
        #
        # Two compounding issues:
        #   1. V_ph is the terminal voltage, not the voltage at the parallel node.
        #      The actual rotor-path impedance needs V_parallel = V_ph - I_FL*(R1+jX1).
        #   2. I_0 (no-load phasor) is not the magnetising phasor at full load —
        #      the voltage drop across R1+jX1 at full load changes both magnitude
        #      and phase of the magnetising current.
        #
        # A correct estimate requires iterative refinement: after solving alpha,
        # re-compute Xtot from the now-known X1 and Xm.
        pf = max(min(self.case.PF_FL, 1.0), -1.0)
        phi = math.acos(pf)
        I_fl = cmath.rect(self.I_FL_ph, -phi)
        I_w = self.case.P_0 / max(3.0 * self.V_ph, EPS)
        I_m = math.sqrt(max(self.I_0_ph**2 - I_w**2, 0.0))
        I_sh = complex(I_w, -I_m)
        I_sr = I_fl - I_sh
        Z_sr = self.V_ph / I_sr
        self._seed_meta = {'I_fl': I_fl, 'I_sh': I_sh, 'I_sr': I_sr, 'Z_sr': Z_sr}
        return max(Z_sr.imag, 1e-6)

    @staticmethod
    def _zpar(z1: complex, z2: complex) -> complex:
        return 1.0 / (1.0 / z1 + 1.0 / z2)

    def _predict(self, alpha: float, R2: float):
        X1 = alpha * self.Xtot
        X2 = (1.0 - alpha) * self.Xtot
        X3 = self.XLR - X1
        if X3 <= 0.0:
            return None

        # True magnetising reactance: Xm = X0 - X1 (X0 is the total no-load
        # reactance, X1 = alpha * Xtot is fixed in this step). This removes the
        # former X1 double-count: Z0 = R1 + j(X1 + Xm) = R1 + j X0.
        Xm = self.X0 - X1
        if Xm <= 0.0:
            return None

        Z0 = complex(self.R1_hot, X1 + Xm)
        I0 = abs(self.V_ph / Z0)
        p_fw = self.case.P_FW if self.case.P_FW is not None else 0.0
        p_core = self.case.P_core if self.case.P_core is not None else max(self.case.P_0 - 3.0 * self.I_0_ph**2 * self.R1_hot - p_fw, EPS)
        P0 = 3.0 * I0**2 * self.R1_hot + p_core + p_fw

        # full-load with shunt branch included
        Zm = complex(0.0, Xm)
        Zr = complex(R2 / max(self.s_FL, EPS), X2)
        Zpar = self._zpar(Zm, Zr)
        Zin = complex(self.R1_hot, X1) + Zpar
        IFL = abs(self.V_ph / Zin)
        PF = max(min(Zin.real / abs(Zin), 1.0), 0.0)
        # 3-phase input power from per-phase quantities (valid for Y and Δ).
        Pin = 3.0 * self.V_ph * IFL * PF
        ETA = self.case.P_out * HP_TO_WATT / max(Pin, EPS)
        Vnode = self.V_ph * (Zpar / Zin)
        Ir = abs(Vnode / Zr)
        TFL = 3.0 * (Ir**2) * (R2 / max(self.s_FL, EPS)) / max(self.omega_s, EPS)

        # LR / BD auxiliary quantities retained for reporting
        Rlr = self.R1_hot + self.R3
        Zlr = abs(complex(Rlr, self.XLR))
        ILR = self.V_ph / max(Zlr, EPS)
        TLR = (3.0 / max(self.omega_s, EPS)) * self.V_ph**2 * self.R3 / max(Zlr**2, EPS)
        # KNOWN-UNKNOWN: Simplified-circuit breakdown-slip prediction (neglects Xm path).
        # Inherits any bias in Xtot from the shunt-correction seed.
        sBDm = R2 / max(math.sqrt(self.R1_hot**2 + self.Xtot**2), EPS)

        return {
            'R2': R2,
            'X1': X1,
            'X2': X2,
            'X3': X3,
            'Xm': Xm,
            'R3': self.R3,
            'I0': I0,
            'P0': P0,
            'IFL': IFL,
            'PF': PF,
            'ETA': ETA,
            'TFL': TFL,
            'ILR': ILR,
            'TLR': TLR,
            'sBD': sBDm,
        }

    def fit(self):
        # KNOWN-UNKNOWN: The objective mixes six error terms; the minimiser may
        # trade off errors rather than converge to a physically meaningful solution.
        # Since Xtot and X0 are fixed, solve only R2 for each alpha and choose best alpha.
        # (Xm = X0 - X1 is resolved inside _predict for each alpha.)
        tols = {
            'I0': max(0.05 * self.I_0_ph, 1e-6),
            'P0': max(0.05 * self.case.P_0, 1e-6),
            'TFL': max(0.05 * self.T_FL, 1e-6),
            'IFL': max(0.03 * self.I_FL_ph, 1e-6),
            'PFFL': max(0.01 * self.case.PF_FL, 1e-6),
            'ETAFL': max(0.01 * self.case.eta_FL, 1e-6),
        }

        best = None
        # KNOWN-UNKNOWN: Alpha grid is coarse (6 discrete values) with uneven
        # spacing. The true X1/X2 split may lie far from any candidate.
        for alpha in [0.30, 0.35, 0.39, 0.40, 0.45, 0.50]:
            def obj_r2(R2: float):
                pred = self._predict(alpha, R2)
                if pred is None:
                    return 1e12
                r = {
                    'I0': (pred['I0'] - self.I_0_ph) / tols['I0'],
                    'P0': (pred['P0'] - self.case.P_0) / tols['P0'],
                    'TFL': (pred['TFL'] - self.T_FL) / tols['TFL'],
                    'IFL': (pred['IFL'] - self.I_FL_ph) / tols['IFL'],
                    'PFFL': (pred['PF'] - self.case.PF_FL) / tols['PFFL'],
                    'ETAFL': (pred['ETA'] - self.case.eta_FL) / tols['ETAFL'],
                }
                return sum(v * v for v in r.values())

            # KNOWN-UNKNOWN: R2 bounds [1e-6, 0.2] hardcoded and motor-size-dependent.
            # Fallback R2 = 0.05 when SciPy unavailable is arbitrary.
            if _HAS_SCIPY:
                res = minimize_scalar(obj_r2, bounds=(1e-6, 0.2), method='bounded')
                R2_best = float(res.x)
                score = float(res.fun)
            else:
                R2_best = 0.05
                score = obj_r2(R2_best)

            pred = self._predict(alpha, R2_best)
            if best is None or score < best['score']:
                best = {'alpha_best': alpha, 'score': score, **pred}

        best['R1_hot'] = self.R1_hot
        best['R1_cold'] = self.case.R1_cold
        best['X1_plus_X2'] = self.Xtot
        best['Xsum_seed_from_shunt'] = self._seed_meta['Z_sr'].imag
        best['Xsum_seed_realpart'] = self._seed_meta['Z_sr'].real
        best['I_fl_complex'] = self._seed_meta['I_fl']
        best['I_sh_complex'] = self._seed_meta['I_sh']
        best['I_sr_complex'] = self._seed_meta['I_sr']
        return best

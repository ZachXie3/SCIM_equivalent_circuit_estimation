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
    s_BD: float
    R_1: float    # raw measured stator phase resistance at ambient
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
    - Xm from no-load (§5): returns X_nl = X1_true + Xm_true, not Xm alone.
      The no-load prediction then double-counts X1, biasing I0 and P0.
    - R3 from LR torque (§6): disconnected from R2 (dual rotor-resistance model).
      LR predictions (ILR, TLR) are exact by construction, providing zero constraint.
    - Xtot from shunt correction (§8): overestimates X1+X2 by (I_mag/I_r)·X1 (~10-25%).
    - Alpha grid: coarse, uneven (6 values). R2 bounds: hardcoded [1e-6, 0.2].

    Logic (may change once unknowns are resolved):
    - Fix Xtot = X1 + X2 from the full-load-with-shunt-current formula.
    - Fix Xm from no-load test.
    - Compute R3 from LR torque/current relation.
    - Use XLR from LR current relation.
    - Iterate alpha only (outer scan), while solving R2 for each alpha.
    """

    def __init__(self, case: MotorCase):
        self.case = case
        self.V_ph = self._per_phase_voltage()
        self.n_s = 120.0 * case.f / case.P
        self.omega_s = 2.0 * math.pi * self.n_s / 60.0
        self.s_FL = (self.n_s - case.n_FL) / self.n_s
        self.R1_hot = self._correct_r1_to_hot()
        self.T_FL = case.P_out * HP_TO_WATT / max(2.0 * math.pi * case.n_FL / 60.0, EPS)
        self.T_LR_Nm = case.T_LR * self.T_FL
        self.T_BD_Nm = case.T_BD * self.T_FL
        self.Xm = self._estimate_xm_from_no_load()
        self.R3 = self._estimate_r3_from_locked_rotor()
        self.XLR = self._estimate_xlr_from_locked_rotor()
        self.Xtot = self._estimate_xtot_from_full_load_with_shunt()

    def _per_phase_voltage(self) -> float:
        c = self.case.connection.upper().strip()
        if c == 'Y':
            return self.case.V_LL / math.sqrt(3.0)
        if c in {'D', 'DELTA', 'Δ'}:
            return self.case.V_LL
        raise ValueError(f'Unsupported connection: {self.case.connection!r}')

    def _correct_r1_to_hot(self) -> float:
        ta = self.case.T_ambient_C
        tb = ta + self.case.temp_rise_C
        return self.case.R_1 * (tb + K1_COPPER) / (ta + K1_COPPER)

    def _estimate_xm_from_no_load(self) -> float:
        # KNOWN-UNKNOWN: This returns X_nl = X1_true + Xm_true (total no-load
        # reactance), not Xm alone. In _predict the no-load impedance becomes
        # R1 + j(X1_guess + X_nl) = R1 + j(X1_guess + X1_true + Xm_true), which
        # double-counts X1 and underestimates I0. The fix requires iterative
        # separation: start with Xm = X_nl, solve, then update Xm = X_nl - X1,
        # re-solve until stable.
        s0 = math.sqrt(3.0) * self.case.V_LL * self.case.I_0
        q0 = math.sqrt(max(s0**2 - self.case.P_0**2, 0.0))
        return 3.0 * self.V_ph**2 / max(q0, EPS)

    def _estimate_r3_from_locked_rotor(self) -> float:
        # KNOWN-UNKNOWN: R3 is the standstill rotor resistance derived from LR torque,
        # but it is disconnected from R2 (running rotor resistance). No skin-effect
        # model bridges the two. Because R3 and XLR are back-solved from LR data,
        # ILR and TLR predictions always exactly match measured values — they provide
        # zero constraint on the fit. A frequency-dependent rotor resistance model
        # would unify R3 and R2.
        return self.T_LR_Nm * self.omega_s / max(3.0 * self.case.I_LR**2, EPS)

    def _estimate_xlr_from_locked_rotor(self) -> float:
        zlr = self.V_ph / max(self.case.I_LR, EPS)
        return math.sqrt(max(zlr**2 - (self.R1_hot + self.R3)**2, EPS))

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
        I_fl = cmath.rect(self.case.I_FL, -phi)
        I_w = self.case.P_0 / max(3.0 * self.V_ph, EPS)
        I_m = math.sqrt(max(self.case.I_0**2 - I_w**2, 0.0))
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

        # KNOWN-UNKNOWN: Z0 = R1_hot + j(X1 + self.Xm) double-counts X1 because
        # self.Xm is X_nl = X1_true + Xm_true (total no-load reactance), not Xm alone.
        # Correct would be: first solve, then Xm = X_nl - X1, re-solve.
        # The I0 and P0 predictions below are therefore under-estimated.
        Z0 = complex(self.R1_hot, X1 + self.Xm)
        I0 = abs(self.V_ph / Z0)
        p_fw = self.case.P_FW if self.case.P_FW is not None else 0.0
        p_core = self.case.P_core if self.case.P_core is not None else max(self.case.P_0 - 3.0 * self.case.I_0**2 * self.R1_hot - p_fw, EPS)
        P0 = 3.0 * I0**2 * self.R1_hot + p_core + p_fw

        # full-load with shunt branch included
        Zm = complex(0.0, self.Xm)
        Zr = complex(R2 / max(self.s_FL, EPS), X2)
        Zpar = self._zpar(Zm, Zr)
        Zin = complex(self.R1_hot, X1) + Zpar
        IFL = abs(self.V_ph / Zin)
        PF = max(min(Zin.real / abs(Zin), 1.0), 0.0)
        Pin = math.sqrt(3.0) * self.case.V_LL * IFL * PF
        ETA = self.case.P_out * HP_TO_WATT / max(Pin, EPS)
        Vnode = self.V_ph * (Zpar / Zin)
        Ir = abs(Vnode / Zr)
        TFL = 3.0 * (Ir**2) * (R2 / max(self.s_FL, EPS)) / max(self.omega_s, EPS)

        # LR / BD auxiliary quantities retained for reporting
        Rlr = self.R1_hot + self.R3
        Zlr = abs(complex(Rlr, self.XLR))
        ILR = self.V_ph / max(Zlr, EPS)
        TLR = (3.0 / max(self.omega_s, EPS)) * self.V_ph**2 * self.R3 / max(Zlr**2, EPS)
        # KNOWN-UNKNOWN: Simplified-circuit breakdown formula (neglects Xm path).
        # Inherits any bias in Xtot from the shunt-correction seed.
        sBDm = R2 / max(math.sqrt(self.R1_hot**2 + self.Xtot**2), EPS)
        Rbd = self.R1_hot + R2 / max(self.case.s_BD, EPS)
        Zbd = abs(complex(Rbd, self.Xtot))
        TBD = (3.0 / max(self.omega_s, EPS)) * self.V_ph**2 * (R2 / max(self.case.s_BD, EPS)) / max(Zbd**2, EPS)

        return {
            'R2': R2,
            'X1': X1,
            'X2': X2,
            'X3': X3,
            'Xm': self.Xm,
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
            'TBD': TBD,
        }

    def fit(self):
        # KNOWN-UNKNOWN: The objective includes I0/P0 which are biased by the Xm
        # double-count in _predict. The optimizer may trade off errors across terms
        # rather than converge to a physically meaningful solution.
        # Since Xtot and Xm are fixed, solve only R2 for each alpha and choose best alpha.
        tols = {
            'I0': max(0.05 * self.case.I_0, 1e-6),
            'P0': max(0.05 * self.case.P_0, 1e-6),
            'TFL': max(0.05 * self.T_FL, 1e-6),
            'IFL': max(0.03 * self.case.I_FL, 1e-6),
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
                    'I0': (pred['I0'] - self.case.I_0) / tols['I0'],
                    'P0': (pred['P0'] - self.case.P_0) / tols['P0'],
                    'TFL': (pred['TFL'] - self.T_FL) / tols['TFL'],
                    'IFL': (pred['IFL'] - self.case.I_FL) / tols['IFL'],
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
        best['R1_raw'] = self.case.R_1
        best['X1_plus_X2'] = self.Xtot
        best['Xsum_seed_from_shunt'] = self._seed_meta['Z_sr'].imag
        best['Xsum_seed_realpart'] = self._seed_meta['Z_sr'].real
        best['I_fl_complex'] = self._seed_meta['I_fl']
        best['I_sh_complex'] = self._seed_meta['I_sh']
        best['I_sr_complex'] = self._seed_meta['I_sr']
        return best

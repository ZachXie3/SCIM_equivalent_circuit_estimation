"""Tasks 3 & 4 — Tests for the standalone R2-from-R3 model and the
Stage 3/4 iterative estimator in ``equivalent_circuit.py``.

Run from the repo root:
    python -m pytest typical_value/test/test_equivalent_circuit.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_TV = Path(__file__).resolve().parent.parent
_ROOT = _TV.parent
sys.path.insert(0, str(_TV))
sys.path.insert(0, str(_ROOT))

import equivalent_circuit as e  # noqa: E402

RESULTS_CSV = _TV / "data" / "eq_parameters.csv"


def _make_case() -> e.MotorCase:
    df = pd.read_csv(RESULTS_CSV)
    r = df[(df["HorsePower"] == 60) & (df["PoleSpeed"] == 4) & (df["Voltage"] == 460)].iloc[0]
    return e.MotorCase(
        V_LL=460.0,
        f=60.0,
        P=4,
        n_FL=float(r["RPM"]),
        P_out=float(r["HorsePower"]),
        I_FL=float(r["Amps"]),
        PF_FL=0.87,
        eta_FL=0.93,
        I_LR=6.0 * float(r["Amps"]),
        T_LR=1.5,
        T_BD=2.2,
        R1_cold=0.15,
        I_0=0.3 * float(r["Amps"]),
        P_0=1.0,
        J=1.0,
    )


# ---------------------------------------------------------------------------
# Standalone R2-from-R3 model (plan.md Stage 2.4 / 2.5)
# ---------------------------------------------------------------------------


def test_r2r3_model_d_reproduces_report_ratio():
    """4-pole, 60 HP, s=1.11% -> R2/R3 = exp(2.277) * s^0.634 * HP^-0.129."""
    R3, s_fl, poles, hp = 0.132, 0.0111, 4, 60.0
    out = e.estimate_r2_from_r3(R3, s_fl, poles, hp)
    expected_ratio = np.exp(2.277) * s_fl ** 0.634 * hp ** -0.129
    assert out["ratio"] == pytest.approx(expected_ratio, rel=1e-3)
    assert out["r2"] == pytest.approx(R3 * expected_ratio, rel=1e-3)
    assert out["pole_match"] is True
    assert out["out_of_range"] is False
    assert out["clamped"] is False


def test_r2_model_falls_back_for_unseen_pole():
    out = e.estimate_r2_from_r3(0.132, 0.0111, 12, 60.0)
    assert out["pole_match"] is False
    fb = out["r2"] / 0.132
    assert fb == pytest.approx(np.exp(3.0090) * 0.0111 ** 0.9239, rel=1e-3)


def test_r2_model_flags_out_of_range():
    out = e.estimate_r2_from_r3(0.132, 0.5, 4, 60.0)
    assert out["out_of_range"] is True
    assert out["clamped"] is True


def test_r2_model_clamps_to_r3():
    out = e.estimate_r2_from_r3(0.132, 0.5, 4, 1.0)
    assert out["clamped"] is True
    assert out["r2"] == pytest.approx(0.132)


def test_r2_model_zero_slip_yields_zero():
    out = e.estimate_r2_from_r3(0.132, 0.0, 4, 60.0)
    assert out["r2"] == 0.0


# ---------------------------------------------------------------------------
# Stage 3/4 iterative estimator
# ---------------------------------------------------------------------------


def test_estimator_converges():
    est = e.EquivalentCircuitEstimator(_make_case())
    out = est.fit()
    assert out["converged"] is True
    assert 1 <= out["outer_iterations"] <= e.MAX_OUTER_ITER
    assert np.isfinite(out["solver_score"])


def test_estimator_physical_ordering():
    est = e.EquivalentCircuitEstimator(_make_case())
    out = est.fit()
    assert 0 < out["R1_cold"] < out["R1_hot"]
    assert 0 < out["R2"] < out["R3"]
    assert min(out["X1"], out["X2"], out["Xm"], out["X3"]) > 0
    assert out["Xm"] > out["X1"]
    assert out["X3"] > 0
    assert 0 < out["s_fl"] < 1


def test_estimator_matches_full_load_current_and_torque():
    est = e.EquivalentCircuitEstimator(_make_case())
    out = est.fit()
    r_final = est._residuals_x1x2(
        np.array([out["X1"], out["X2"]]), out["R2"], out["Xm"]
    )
    assert abs(r_final[0]) < 0.01
    assert abs(r_final[1]) < 0.01
    assert abs(r_final[2]) < 0.02


def test_estimator_diagnostic_ratios():
    est = e.EquivalentCircuitEstimator(_make_case())
    out = est.fit()
    assert out["R2_R3_ratio"] == pytest.approx(out["R2"] / out["R3"])
    assert out["Xm_X1_ratio"] == pytest.approx(out["Xm"] / out["X1"])
    assert np.isclose(out["Xtot"], out["X1"] + out["X2"])
    assert np.isclose(out["XLR"], out["X1"] + out["X3"])


def test_estimator_warns_on_unphysical_slip():
    case = _make_case()
    case.n_FL = 1900.0  # above synchronous speed -> negative slip
    est = e.EquivalentCircuitEstimator(case)
    out = est.fit()
    assert "slip_out_of_range" in out["warning_flags"]
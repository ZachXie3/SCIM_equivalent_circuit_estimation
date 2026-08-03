"""Regression test using EquivalentCircuitParameters Rev.5 output for 734250BQ_Rev2."""

import sys
from math import isclose
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from examples_calculator import calculate_row

# Inputs are mapped from 734250BQ_Rev2 DesignSheet and the calculated BDT ratio.
# Important: the Rev.5 workbook's text formula extracts NoLoadAmps as 07.93 from
# "77.45 / 107.93". This test intentionally keeps NoLoadAmps=7.93 to match the workbook output exactly.
CASE_734250BQ = {
    "PoleSpeed": 4,
    "Voltage": 460,
    "Frequency": 60,
    "Amps": 295.13,
    "NoLoadAmps": 7.93,
    "Connection": "Delta",
    "WindingResistAt105": 1.5537741546965441e-2,
    "BarRes": 1.4435179531574249e-2,
    "ERRes": 3.8970000316449998e-3,
    "BDTCorelFactor": 283.0 / 214.4,
    "ReactCoilEndRun": 12.265219331483715,
    "ResTotalStart": 0.11354942619800568,
    "ReactTotalRun": 0.71941018104553223,
    "ReactStatorSlotRun": 17.003923416137695,
    "ReactStatorSlotStart": 17.003923416137695,
    "ReactRotorSlotRun": 75.602973937988281,
    "ReactRotorSlotStart": 34.203765869140625,
    "ReactZigZagRun": 15.56425952911377,
    "ReactZigZagStart": 13.945422172546387,
    "ReactBeltLeakRun": 8.2987122177961119,
    "ReactTotalStart": 0.47418895363807678,
}

EXPECTED = {
    "Zbase": 0.89987843942407708,
    "R1": 5.1792471823218136e-3,
    "R2": 6.1107265210730835e-3,
    "R3": 3.2670561550346741e-2,
    "X1": 5.2373752819579435e-2,
    "X2": 0.12930062034433357,
    "X3": 9.2416130550312378e-2,
    "Xm": 33.490683963080436,
    "XdPrime": 0.18117709005600605,
    "XdDoublePrime": 0.1061823355290939,
    "ShortCircuitTimeConstant": 7.8826850352323341e-2,
    "SubTransientTimeConstant": 1.0526515122012352e-2,
    "OpenCircuitTimeConstant": 14.593973167983018,
    "L1": 1.3892569415848598e-4,
    "L2": 3.4298054787748627e-4,
    "L3": 2.4514139997513563e-4,
    "FirstCycleInrush": 2.3499994262923631,
}


def test_734250BQ_equivalent_circuit_rev5():
    actual = calculate_row(CASE_734250BQ)
    for key, expected in EXPECTED.items():
        assert isclose(actual[key], expected, rel_tol=1e-10, abs_tol=1e-12), (
            key,
            actual[key],
            expected,
        )


if __name__ == "__main__":
    test_734250BQ_equivalent_circuit_rev5()
    print("PASS: 734250BQ Rev.5 equivalent circuit regression test")

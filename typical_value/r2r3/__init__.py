"""R2/R3 ratio analysis methods (plan.md Stage 2).

Each method estimates the running/standstill rotor resistance ratio
``R2/R3`` from nameplate-available quantities and is scored on
``data/eq_parameters.csv``. Methods live in one module each and share a
uniform fit / predict / score interface so a new method can be added without
touching the report generator:

    * ``models.py``        — power-law models A-D (``fit_model_*``,
                             ``predict_model_*``, ``score_model``)
    * ``constant_bin.py``  — constant ``R2/R3`` per slip bin (approved table
                             method, plan.md §2.7 / §11.1) plus the grouping-
                             constant exploration (§6)
    * ``report.py``        — renders ``reports/r2r3_report.md``

Adding a second analysis method:

    1. Create ``r2r3/<method>.py`` exposing ``fit(df) -> params``,
       ``predict(params, df) -> pd.Series`` (predicted R2) and
       ``score(df) -> dict`` (metrics), mirroring ``constant_bin.py``.
    2. Wire the new section into ``report.py::build_markdown``.

Data lives in ``../data/eq_parameters.csv``; the estimator consumes the
adopted prior via ``equivalent_circuit.py``.
"""

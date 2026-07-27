# Spec 01 Equivalent Circuit Estimation — Updated `Xtot` Seed Integrated into Estimator

## Update summary

The estimator now uses a **full-load-with-shunt-current** seed for total leakage reactance:

1. Build the full-load current phasor from `I_FL` and `PF_FL`
2. Estimate the shunt-branch current from no-load test data `I_0`, `P_0`
3. Subtract the shunt current from full-load current to obtain the effective current through the series / rotor path
4. Compute the corresponding branch impedance
5. Use the **imaginary part** of that impedance as the improved seed for
\[
X_{tot} = X_1 + X_2
\]

This replaces the earlier series-only seed:
\[
Z_{FL} = \frac{V_{ph}}{I_{FL}},\qquad X_{tot} = \sqrt{Z_{FL}^2 - R_{tot}^2}
\]
which underestimated `X1 + X2` by ignoring the magnetizing branch at full load.

## New seed formula

### Full-load current phasor
\[
\phi_{FL} = \cos^{-1}(PF_{FL}),\qquad
\underline I_{FL} = I_{FL}(\cos\phi_{FL} - j\sin\phi_{FL})
\]

### Shunt current from no-load data
\[
I_w = \frac{P_0}{3V_{ph}}
\]
\[
I_m = \sqrt{I_0^2 - I_w^2}
\]
\[
\underline I_{sh} \approx I_w - jI_m
\]

### Effective series / rotor-path current
\[
\underline I_{sr} = \underline I_{FL} - \underline I_{sh}
\]

### Effective branch impedance
\[
\underline Z_{sr} = \frac{V_{ph}}{\underline I_{sr}}
\]

### Improved leakage seed
\[
X_{tot,seed} = \Im(\underline Z_{sr})
\]

## Sample-1 direct seed check

For the uploaded sample, the direct full-load-with-shunt-current estimate gives:
\[
X_1 + X_2 \approx 2.8098\ \Omega
\]
versus IEEE 112 Method F1 ground truth:
\[
X_1 + X_2 = 2.7266\ \Omega
\]
which is only about `+3.05%` high.

## Important observation from full estimator run

Even after integrating this better seed, the **full optimizer** still converges to a lower total leakage solution because the rest of the objective structure and the locked-rotor-based bounds still pull the solution downward. So the seed issue is improved, but the total estimator still remains leakage-biased low.

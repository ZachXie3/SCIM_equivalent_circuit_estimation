# Cluster + NN R2/R3 Estimation Plan

Status: draft (review in progress). Fold in the review answers before implementation.

## 1. Objective

Predict the running/standstill rotor resistance ratio `R2/R3` from
nameplate-available features using **weighted clustering**, where:

- each feature carries a learned weight (engineering: features have very
  different importance — clustering with equal weights is wrong);
- the prediction for a rating is the (log-space) average `R2/R3` of the
  cluster it falls into;
- a small NN (a soft-assignment network) learns the feature weights so the
  prediction error (RMSE) is minimised, while a sparsity penalty prefers
  fewer features.

This is a second, independent analysis alongside `r2r3/` (power-law models
and the constant-per-bin table). The eventual output is a
cluster → `R2/R3` lookup table usable by the estimator, like the slip-segment
table in `r2r3/constant_bin.py`.

## 2. Data

- Source: `typical_value/data/eq_parameters.csv` (4195 rows).
- Target: `R2/R3` (modelled in **log space**: `y = ln(R2/R3)`).
- Dropped rows (in order):
  1. `slip <= 0` — 3 rows (2 zero-slip, 1 negative).
  2. `R2/R3 >= 1.0` — 339 rows (338 floating-point ties `1.000000..1.000056`,
     almost all shallow-bar AL `Diamond`/`Oval` designs + 2 `Dbl - Closed`).
     **Decision (user): these are false data, drop them.** The estimator
     already clamps to `R2 <= R3`; the remaining data has `R2/R3 < 1`.
     - Engineering note: the dropped rows are the near-`R2 = R3` shallow-bar
       population, so the fitted model will not predict ratios near 1 for
       shallow-bar motors — acceptable per the decision, but worth re-examining
       if shallow-bar predictions are ever needed.
- Kept rows for fitting: **3853** (ratio `0.071 .. <1.0`, mean ≈ 0.37).

### 2.1 Features (allowed input)

| Feature | Type | Encoding |
|---|---|---|
| `PoleSpeed` | numeric (2/4/6/8) | `z(PoleSpeed)` — 1 dim |
| `slip = (SRPM − RPM)/SRPM` | numeric | `z(slip)` — 1 dim |
| `HorsePower` | numeric (0.4–4600) | `z(ln HP)` — 1 dim |
| `NemaDesign` | **ordinal** | `none=0, A=1, B=2, C=3` — 1 dim |
| `NemaKVACode` | **ordinal** (alphabetical) | one level per letter, rank 0..13 (see §2.3) — 1 dim |
| `BarMaterial` (= `RotorMaterial`) | categorical | one-hot {AL, CU, Special} — 3 dims |
| `Name` (= `EfficiencyLevel`) | categorical | one-hot {4 levels} — 4 dims |

Total encoded dims = 12, feature groups = 7, learned weights = 7.

Note: `SRPM == 120·f/Poles` exactly in this file, so `slip` matches the
`r2r3` analysis slip. `HorsePower` is included per decision (it was the
2nd-strongest predictor in the prior study).

### 2.2 NemaDesign ordinal mapping

| Raw level | Map to | value |
|---|---|---|
| `A,B,C` | C | 3 |
| `A,B` | B | 2 |
| `A` | A | 1 |
| (missing) | none | 0 |
| `C` | C | 3 |

(1042 missing rows become `none` — keep, don't drop.)

### 2.3 NemaKVACode — ordinal, one level per letter

**Decision: each present letter is its own group** — do not merge rare
letters. The letters are ordered alphabetically, so the distance between two
levels reflects their alphabetical separation (e.g. `C` is farther from `G`
than from `D`). Encoded as a single ordinal numeric (compressed alphabetical
rank), then standardized:

| Ordinal | Letter | n |
|---:|---|---:|
| 0 | C | 6 |
| 1 | D | 61 |
| 2 | E | 178 |
| 3 | F | 597 |
| 4 | G | 1947 |
| 5 | H | 610 |
| 6 | J | 303 |
| 7 | K | 204 |
| 8 | L | 159 |
| 9 | M | 102 |
| 10 | N | 15 |
| 11 | P | 4 |
| 12 | R | 1 |
| 13 | V | 6 |

- Any monotone mapping preserves the ordering; compressed rank is used.
- 2 missing `NemaKVACode` rows → impute to the modal letter `G` (ordinal 4)
  or drop (only 2 rows; pick in prep).

## 3. Method

### 3.1 Weighted distance and soft assignment

- Per-feature weight `w_g ∈ (0,1)`, `w_g = sigmoid(z_g)` (z = trainable).
- Distance from row `i` to cluster centroid `c`, over feature groups:

      d_ic = Σ_g  w_g · ‖x_{i,g} − μ_{c,g}‖²

  For one-hot groups this is weighted Hamming; for numeric/ordinal groups it
  is weighted squared difference on standardized values — comparable scale.

- Soft assignment (temperature τ):

      a_ic = softmax(−d_ic / τ)

### 3.2 Cluster parameters — deterministic, EM-style (NOT NN parameters)

Given the current weights, recompute each step:

      μ_{c,g} = Σ_i a_ic · x_{i,g} / Σ_i a_ic      (weighted centroid)
      m_c     = Σ_i a_ic · y_i / Σ_i a_ic          (weighted log-ratio mean)

### 3.3 Prediction and loss

- Prediction (soft, log space):  ŷ_i = Σ_c a_ic · m_c
- Loss:

      L = RMSE_train( y − ŷ ) + β · count(active features) · rmse_ref

  - `RMSE` in log space = relative error on the ratio.
  - **Feature-count penalty** (decision 4): a feature is **active** when its
    learned weight is not 0 (`w > ACTIVE_W = 0.05`); the penalty is
    `β · (number of active features) · rmse_ref`, where `rmse_ref` is a fixed
    reference (the RMSE of a constant predictor) so the count term is measured
    in the same units as RMSE.
  - `β` = the cost, in RMSE percentage points, of keeping one feature. A
    feature is retained only if removing it would cost more than `β` pp.
  - The count is implemented with a smooth surrogate
    `count ≈ Σ (1 − exp(−k·w_g))` so it stays differentiable; `β` is tuned on
    the validation split (β-sweep: fewest active features within `tol_pp` of
    the best).
  - **Weights are normalised** (`w = softmax(z)·G`, so `Σ w = G`): uniform
    shrinking is impossible, so the penalty can only be reduced by actually
    zeroing a feature (no degenerate "shrink everything" solution).
  - The delivered model is refit on the **active features only**.

### 3.4 The NN updates weights ONLY (per decision)

- Learnable parameters: **z (7 scalars)** only.
- Centroid `μ` and mean `m` are recomputed deterministically from the soft
  assignments every step (3.2) — they are derived, not NN parameters.
- Optimizer: Adam on `z`. Defaults to tune on validation: `lr ≈ 1e-2`,
  ~500 steps, **fixed `τ`** (start `0.5`), a few restarts. Init even weight
  `w_g = 1` (`z = 0`, uniform softmax).
- τ annealing (`1.0 → 0.1`) is a **future improvement**, not part of the first
  implementation (see §8).
- Early stop on **validation** RMSE.

### 3.5 Why gradient-based soft assignment, not RL

Question: is RL better for updating the weights? **No** — use the
differentiable soft-assignment model:

- **Easy to implement:** the soft model gives exact gradients, so it is a
  ~100-line numpy Adam loop. RL (policy gradient / bandit) needs reward
  scaling, exploration, baselines, many samples, and is finicky to tune.
- **Logically sound:** the objective (RMSE + sparsity) is deterministic and
  differentiable through the soft assignment; RL throws away that structure
  and estimates gradients by sampling.
- **Fits the dataset:** 3853 rows, 7 weights — a tiny, well-conditioned
  optimisation. RL's variance is pure noise here.
- RL would only be justified if the assignment were non-differentiable (hard
  KMeans); the soft formulation removes that need.

### 3.6 Final (delivered) model is a hard lookup

- Train with soft assignment to learn `w`.
- **Feature pruning:** after `γ` is chosen, the delivered model is refit using
  **only the active features** (weight > 0.5): inactive features are excluded
  from the encoding before building the lookup, so the delivered table truly
  uses the minimal feature set.
- **Deliverable:** hard assignment (`argmin_c d_ic`, τ→0) with the learned
  `w` → cluster table `{cluster → ratio = exp(m_c)}`, `m_c` = mean `y` of the
  hard cluster, built on the **train** split (kept identical to the fitted
  split so the train/validation/test numbers are directly comparable across
  methods). For production deployment the lookup may be refit on
  train+validation before shipping — but the reported numbers always come from
  the train-fitted version.
- Prediction for any rating: nearest-cluster ratio, clamped to `(0, 1)`.
- Report metrics on this delivered hard model (not the soft training proxy),
  so the reported numbers match what a user actually gets.

## 4. Splits & evaluation

- **Three-way split** (decision): `train 70% / validation 15% / test 15%`,
  fixed seed. **Stratify by coarse `R2/R3` target bins** (e.g., quartiles) to
  keep the ratio distribution similar across splits.
  `PoleSpeed` and `BarMaterial` are **not** used as stratification keys (they
  are irrelevant for that purpose and should not be combined there); they
  remain individual model features and their relevance is judged by the
  feature-count penalty / ablation.
- Standardization (`z`) fit on **train only**, applied to val/test.
- `validation` → early stopping and `γ` selection (γ-sweep).
- `test` → **used exactly once** for the final model performance.
- **Fair comparison:** every method (cluster, cluster-unweighted, slip+HP
  power law, per-pole Model D, slip-segment table) is **fitted on the train
  split only**, then scored on train / validation / test. Baselines are *not*
  fitted on train+validation (that would give them more data than the cluster
  model).
- Metrics (same convention as `r2r3/`): RMSE, MAE, P10/P50/P90 of relative
  error on `R2/R3`, reported for **train, validation, and test** per method.
- Note: the cluster model's validation number is mildly optimistic (validation
  selects its weights); baselines have no selection step. The test numbers are
  unbiased for all methods.
- Feature ablation: report ΔRMSE from removing each feature (direct check of
  "does this feature earn its keep") alongside the learned weights.

## 5. Cluster-count sweep (later, not now)

- Loop `K = 4..8`, plot **test** RMSE vs K (train+test on the same axes).
- Choose the smallest K within ~1 pp of the best (simplicity favoured).
- In-sample RMSE always drops with K (it is within-cluster variance), so the
  sweep must use test RMSE — this is where "is more clusters worth it" is
  judged.

## 6. Deliverables

- `typical_value/cluster/cluster-plan.md` — this plan.
- `typical_value/cluster/cluster_model.py` — reusable module:
  `fit(df) -> params` (weights, centroids, cluster means),
  `predict(params, df) -> Series` (predicted ratio), plus a
  `save`/`load` for the lookup table.
- `typical_value/cluster/cluster_analysis.py` — end-to-end runner:
  load → prep → split → train → evaluate → write report + plots.
- `typical_value/reports/cluster_report.md` — data-prep summary, learned
  weights, feature ablation, cluster table (feature bounds, n, predicted
  ratio, within-cluster RMSE), train/val/test metrics vs baselines,
  K-sweep table.
- `typical_value/reports/plots/` — weight bar chart, RMSE-vs-K (train+test),
  predicted-vs-actual parity (matplotlib approved; add to venv).
- `typical_value/reports/cluster_lookup.csv` — the final
  cluster → `R2/R3` lookup (deliverable for the estimator).
- tests in `typical_value/tests/` following the existing pattern
  (e.g., `test_cluster_model.py`: prep counts, split sizes, fit/predict
  shapes, weight bounds, clamps, reproducibility with fixed seed).

## 7. Open questions / to confirm during implementation

1. **`NemaKVACode` missing rows (2)**: impute to modal letter `G`, or drop
   (§2.3).
2. **Lookup applicability for the estimator**: final decision on whether the
   estimator consumes `cluster_lookup.csv` directly (replacing/alongside the
   slip-segment table) is out of scope now.
3. **Ratio ≥ 1 drop** is recorded as decided (§2); re-examine only if
   shallow-bar (near-`R2=R3`) predictions are ever needed.

Decisions already recorded (do not re-open): NemaKVACode one level per letter
(§2.3); feature-count penalty `γ · count(active) · rmse_ref`, `γ` tuned on
validation (§3.3); fixed `τ` (annealing deferred); split stratified by
`R2/R3` target bins, not `PoleSpeed × BarMaterial`; `HorsePower` included;
`R2/R3 ≥ 1` rows dropped.

## 8. Out of scope / future improvements

- RL-based weight learning (rejected, §3.5).
- τ annealing (`1.0 → 0.1`) — listed as a future improvement (§3.4).
- Per-row (attention) weights instead of global feature weights.
- Cluster-count sweep (deferred to §5).
- Additional features beyond §2.1 (e.g., locked-rotor current) — revisit only
  if the ablation shows a feature is missing.
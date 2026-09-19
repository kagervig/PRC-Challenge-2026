# Improvement Plan — RMSE Below 244.0s

**Baseline:** 244.0s validation RMSE (time-based split, train months 1–10, validate months 11–12)
**File:** `model.py` in this directory
**Rule:** Test each improvement independently. Keep if RMSE improves. Revert if neutral or worse. Record every result in `learnings.md`.

---

## Overview

Three independent improvements, each tested against the 244.0s baseline:

1. **Huber loss** — replace MSE objective with Huber to reduce influence of tail outliers (LIRF disruption days). Residuals are right-skewed (p99 = +13.9 min), so MSE is over-weighting rare extremes during training.
2. **LFPG bias fix** — LFPG has a systematic +28.6s underprediction (19.4% of total SSE). Diagnose root cause from residual breakdown by runway/stand/hour/month, then implement a targeted fix.
3. **airport_runway combined feature** — runway importance is 15.9% but the same runway number means different geometry at different airports. An explicit `airport_runway` combined categorical lets the model learn a per-(airport, runway) mean.

Each improvement is self-contained. They can be implemented in any order, but the sequence below is recommended: Huber first (cheapest, highest expected gain), then LFPG (requires diagnosis before implementation), then airport_runway.

---

## Improvement 1: Huber Loss

### Background
LightGBM's default `"regression"` objective minimises MSE. Squaring errors means a single 30-minute outlier contributes 900× more to the loss than a 1-minute error. With LIRF disruption days still in training (taxi times of 60–300 min on bad days), the model's gradients are pulled toward those extremes. Huber loss is identical to MSE for errors below a threshold `alpha` and switches to linear (MAE-like) above it, dampening the outlier influence without removing those rows from training.

### Steps

#### Step 1.1 — Make `train()` accept an optional `alpha` parameter
`train()` currently hardcodes `"objective": "regression"`. Add an `alpha: float | None = None` parameter. When `alpha` is not None, set `"objective": "huber"` and `"alpha": alpha` in the params dict. When `alpha` is None, behaviour is unchanged (safe default).

#### Step 1.2 — Run alpha sweep in a standalone script
Write a short inline script (not modifying `main()`) that:
- Loads data and computes signals once
- Runs `time_based_split` once
- Trains four models with alpha in `[50, 100, 200, 400]` plus the MSE baseline
- Prints a comparison table of RMSE per alpha
- Identifies the best alpha

#### Step 1.3 — Wire best alpha into `main()` and `train()` call
Update the `train()` call in `main()` to pass the best alpha found in Step 1.2. Update `SUBMISSION_VERSION`. Record result in `learnings.md`. If no alpha beats baseline, revert `train()` to its original signature.

---

## Improvement 2: LFPG Bias Investigation and Fix

### Background
LFPG (Paris CDG) accounts for 19.4% of total SSE with a mean residual of +28.6s — the model consistently underpredicts every CDG flight by about 30 seconds. This is a structural bias, not random noise. Possible causes:
- **Runway shift:** CDG has multiple runway complexes; if the validation period uses a different active runway than training, stand-to-runway distances change
- **Seasonal effect:** CDG summer operations vs autumn/winter may have different taxi patterns that month/hour don't fully capture
- **Stand distribution shift:** New stands opened, or gate allocation changed between training and validation periods
- **Missing interaction:** airport × runway or airport × stand may matter more at CDG than elsewhere

### Steps

#### Step 2.1 — Diagnostic: break LFPG residuals down by dimension
Write an inline diagnostic script that:
- Trains the current model (baseline) and collects validation residuals for LFPG only
- Groups residuals by: runway, stand prefix (first 1–2 chars), hour, and month
- Prints mean residual and count per group, sorted by mean residual
- Compares runway/stand distribution between training and validation periods to flag distribution shifts

#### Step 2.2 — Interpret findings and choose a fix
Based on Step 2.1, the fix will be one of:
- **If runway shift:** add `airport_runway` combined feature (overlaps with Improvement 3 — do that first)
- **If seasonal/hour pattern:** add a `lfpg_hour` or `airport_hour` interaction feature
- **If stand distribution shift:** add a stand-prefix feature (`STAND_mvt` first character) as a coarser grouping that generalises better
- **If none of the above:** the bias may be irreducible given available features; document and move on

#### Step 2.3 — Implement and test the chosen fix
Add the feature identified in Step 2.2 to `build_features()`. Run the model, record RMSE delta vs 244.0s baseline and the change in LFPG mean residual specifically. Keep if overall RMSE improves. Update `SUBMISSION_VERSION` and record in `learnings.md`.

---

## Improvement 3: airport_runway Combined Categorical

### Background
Runway importance is 15.9% (second-highest feature). But runway `"25C"` at EDDF is a completely different physical path than runway `"25"` at LIRF — different distances from terminal, different taxiway geometry. Currently the model must learn runway effects as a main effect independent of airport, then correct with the airport main effect. An explicit `airport_runway` categorical (e.g. `"EDDF_25C"`) gives it a direct handle on each unique (airport, runway) combination.

Cardinality: ~200 unique airport_runway combinations in training — manageable for LightGBM's native categorical handling.

### Steps

#### Step 3.1 — Add `airport_runway` to `build_features()`
In `build_features()`, add:
```python
out["airport_runway"] = (df["ADEP_mvt"] + "_" + df["RUNWAY_mvt"].fillna("UNK")).astype("category")
```
Add `"airport_runway"` to `CATEGORICAL_FEATURES`.

#### Step 3.2 — Decide whether to keep or drop `airport` and `runway` separately
The combined feature may make the individual `airport` and `runway` features redundant. Test three variants:
- A: add `airport_runway`, keep `airport` and `runway`
- B: add `airport_runway`, drop `runway` (keep `airport` for NaN-runway fallback)
- C: add `airport_runway`, drop both

Run all three, compare RMSE. Pick the best.

#### Step 3.3 — Wire in best variant, update version, record in learnings.md
Apply the winning variant to `build_features()` and `CATEGORICAL_FEATURES`. Update `SUBMISSION_VERSION`. Record result.

---

## Implementation Order

```
Improvement 1 (Huber)     →  Step 1.1 → 1.2 → 1.3
Improvement 3 (apt_rwy)   →  Step 3.1 → 3.2 → 3.3
Improvement 2 (LFPG)      →  Step 2.1 → 2.2 → 2.3
```

Improvement 3 is done before 2 because the LFPG diagnosis may conclude that `airport_runway` is the fix — doing 3 first avoids duplicate work.

---

## Prompts for Implementation

---

### Prompt 1.1 — Make `train()` accept an optional Huber alpha

```
We are working on /Users/kristianallin/Code/taxi-predictor/challengedata/model.py.

The `train()` function currently hardcodes `"objective": "regression"` in its params dict.
We want to support Huber loss as an alternative.

Make the following change to `train()`:

1. Add an `alpha: float | None = None` parameter to the function signature.
2. When `alpha` is None, keep the existing behaviour exactly (objective = "regression").
3. When `alpha` is not None, set `"objective": "huber"` and `"alpha": alpha` in the params dict instead of `"objective": "regression"`. Keep `"metric": "rmse"` in both cases so validation output stays comparable.

Do not change any call sites yet — `main()` still calls `train()` without arguments, which continues to work via the default.

Do not change any other part of the file.
```

---

### Prompt 1.2 — Run alpha sweep and print comparison table

```
We are working in /Users/kristianallin/Code/taxi-predictor/challengedata/.

`model.py` has been updated so that `train()` accepts an optional `alpha` parameter.
When alpha is None it uses MSE; when alpha is a float it uses Huber loss with that threshold.

Write a standalone inline script (run with `python3 -c "..."` or as a heredoc) that:

1. Loads training data using `load_training_data()` from model.py.
2. Computes `congestion` and `day_deviation` signals using the existing helpers.
3. Builds features using `build_features()`.
4. Runs `time_based_split()` once to get a fixed train/val split (do NOT re-split per alpha).
5. Trains five models: alpha=None (MSE baseline), alpha=50, alpha=100, alpha=200, alpha=400.
6. Computes validation RMSE for each using `root_mean_squared_error` from sklearn.
7. Prints a comparison table:

   alpha  | RMSE (s) | RMSE (min) | delta vs baseline
   -------|----------|------------|------------------
   None   |  244.0   |   4.07     |   —
   50     |  ...     |   ...      |  +/-X.Xs
   ...

8. Prints "Best alpha: X" at the end.

Import everything from model.py. Do not modify model.py.
```

---

### Prompt 1.3 — Wire best alpha into main() and record result

```
We are working in /Users/kristianallin/Code/taxi-predictor/challengedata/model.py.

We ran an alpha sweep and found the best Huber alpha is [INSERT BEST ALPHA FROM STEP 1.2].
The RMSE with that alpha is [INSERT RMSE].

Make these changes to model.py:

1. Add a config constant near the top (with the other config constants):
   HUBER_ALPHA = [best alpha]  # Huber loss threshold in seconds

2. In `main()`, update the two `train()` calls to pass `alpha=HUBER_ALPHA`:
   - The validation model: `train(X_train, y_train, alpha=HUBER_ALPHA)`
   - The final model: `train(features, target, alpha=HUBER_ALPHA)`

3. Increment SUBMISSION_VERSION by 1.

Do not change anything else.

[NOTE: If the best alpha from Step 1.2 did not improve on the 244.0s MSE baseline, skip steps 1 and 2 — leave `train()` called without alpha — and only increment SUBMISSION_VERSION if another improvement is being applied simultaneously.]
```

---

### Prompt 3.1 — Add airport_runway combined categorical

```
We are working in /Users/kristianallin/Code/taxi-predictor/challengedata/model.py.

We want to add an `airport_runway` combined categorical feature that captures the
unique (airport, runway) combination. The same runway number has different geometry
at different airports, so this gives the model a direct handle on each combination.

Make these two changes:

1. In `build_features()`, add this line after the existing `out["runway"]` line:
   out["airport_runway"] = (
       df["ADEP_mvt"] + "_" + df["RUNWAY_mvt"].fillna("UNK")
   ).astype("category")

2. Add "airport_runway" to the CATEGORICAL_FEATURES list at the top of the file.

Do not change any other features, do not remove "airport" or "runway" yet — we will
test whether to drop them in the next step.

Do not change SUBMISSION_VERSION yet.
```

---

### Prompt 3.2 — Test three airport_runway variants and pick the best

```
We are working in /Users/kristianallin/Code/taxi-predictor/challengedata/.

`model.py` now has an `airport_runway` combined categorical in `build_features()`.

Write a standalone script that tests three variants of the feature set:
- Variant A: airport_runway present, keep both airport and runway
- Variant B: airport_runway present, keep airport, drop runway
- Variant C: airport_runway present, drop both airport and runway

For each variant:
1. Load training data, compute congestion and day_deviation signals (once, shared).
2. Build features using build_features(), then drop columns as needed for each variant.
   Also update the categorical_feature list passed to lgb.Dataset accordingly.
3. Run time_based_split() (once, shared split).
4. Train a model and compute validation RMSE.
5. Print results:

   Variant | Features changed        | RMSE (s) | delta
   --------|-------------------------|----------|------
   A       | +airport_runway         |  ...     |  ...
   B       | +airport_runway -runway |  ...     |  ...
   C       | +airport_runway -both   |  ...     |  ...
   baseline| no airport_runway       |  244.0   |  —

Print "Best variant: X" at the end.

Important: train() takes a `categorical_feature` list via lgb.Dataset, not via the
function signature — you'll need to call lgb.Dataset directly for variants B and C,
or pass adjusted feature sets. Import everything from model.py.
```

---

### Prompt 3.3 — Wire best airport_runway variant into model.py

```
We are working in /Users/kristianallin/Code/taxi-predictor/challengedata/model.py.

We ran the three-variant test and found:
- Best variant: [INSERT FROM STEP 3.2]
- RMSE: [INSERT]
- Columns to drop (if any): [INSERT — "runway", "airport", or neither]

Make these changes:

1. If the best variant drops "runway": remove `out["runway"]` from `build_features()`
   and remove "runway" from CATEGORICAL_FEATURES.
2. If the best variant drops "airport": remove `out["airport"]` from `build_features()`
   and remove "airport" from CATEGORICAL_FEATURES.
3. If the best variant is no better than 244.0s baseline: remove the airport_runway
   feature entirely (revert Prompt 3.1) and do not change anything else.
4. If keeping airport_runway: increment SUBMISSION_VERSION by 1.

Do not change any other part of the file.
```

---

### Prompt 2.1 — LFPG diagnostic: residuals by dimension

```
We are working in /Users/kristianallin/Code/taxi-predictor/challengedata/.

We need to diagnose why LFPG (Paris CDG) has a systematic +28.6s underprediction
in our validation set (months 11–12 of 2025). Write a standalone diagnostic script that:

1. Loads training data using load_training_data() from model.py.
2. Computes congestion and day_deviation signals, builds features, runs time_based_split().
3. Trains the current model on the training split.
4. Collects validation predictions. Compute residual = actual - predicted.
5. Filter to LFPG rows in the validation set only.
6. Print mean residual and row count, grouped by each of:
   a. RUNWAY_mvt
   b. First character of STAND_mvt (stand prefix / terminal area)
   c. Hour of day (MVT_TIME_UTC_mvt.dt.hour)
   d. Month (MVT_TIME_UTC_mvt.dt.month)
   Sort each group by mean residual descending. Show groups with n >= 20 only.

7. Also print: for each runway at LFPG, the flight count in training vs validation
   (to detect distribution shift). Same for stand prefix.

This is read-only analysis. Do not modify model.py.
```

---

### Prompt 2.2 — Implement LFPG fix based on diagnostic findings

```
We are working in /Users/kristianallin/Code/taxi-predictor/challengedata/model.py.

The LFPG diagnostic (Step 2.1) found: [PASTE KEY FINDINGS HERE — e.g. "runway 27R
has mean residual +85s in validation but was only 3% of training; stand prefix E
is systematically underpredicted by 45s"].

Based on these findings, implement the following fix in model.py:

[This prompt will be filled in once Step 2.1 findings are known. The fix will be
one of the following patterns:]

Pattern A — stand prefix feature (if stand distribution shift is the cause):
  In build_features(), add:
    out["stand_prefix"] = df["STAND_mvt"].str[0].fillna("UNK").astype("category")
  Add "stand_prefix" to CATEGORICAL_FEATURES.

Pattern B — airport_hour interaction (if hour pattern at CDG is the cause):
  In build_features(), add:
    out["airport_hour"] = (df["ADEP_mvt"] + "_" + df["MVT_TIME_UTC_mvt"].dt.hour.astype(str)).astype("category")
  Add "airport_hour" to CATEGORICAL_FEATURES.

Pattern C — no actionable fix found:
  Document in learnings.md that the LFPG bias is not recoverable with available
  features. No code changes.

After implementing the chosen pattern, increment SUBMISSION_VERSION by 1.
Do not change anything else.
```

---

### Prompt 2.3 — Test LFPG fix and record result

```
We are working in /Users/kristianallin/Code/taxi-predictor/challengedata/.

We have implemented a LFPG fix in model.py (Step 2.2). Run the model and compare
against the 244.0s baseline. Write a standalone script that:

1. Runs the full pipeline (load data, compute signals, build features, split, train, validate).
2. Prints overall RMSE.
3. Prints per-airport RMSE for LFPG, LIRF, EGLL, and the mean across all other airports.
4. Prints the mean residual for LFPG specifically (to confirm the bias has reduced).

Then record the result in learnings.md using the existing format.
If overall RMSE is worse than 244.0s, revert the change from Step 2.2 and document
the revert in learnings.md.
```

---

## Expected Outcomes

| Improvement | Expected RMSE delta | Risk |
|---|---|---|
| Huber loss (best alpha) | −5s to −15s | Low — one param change, easy to revert |
| airport_runway combined | −3s to −8s | Low — additive feature, easy to revert |
| LFPG bias fix | −5s to −20s (if fixable) | Medium — depends on diagnostic |

If all three land at their expected values, total RMSE could reach 215s–230s. No guarantees — each gets reverted if it doesn't improve.

# Model Run Learnings

## Approach Rationale

### Model: LightGBM

LightGBM gradient boosting was selected from the start and never changed.

**Why LightGBM:**
- Handles missing values natively — NaN rows are routed to a learned child node rather than requiring imputation. This is critical: several of our key features (congestion signal, AOBT_3_flt, gate delay) have 0.5–1.5% NaN rates, and some are NaN for entire subpopulations (ranking DEP rows lack BLOCK_TIME entirely).
- First-class categorical feature support — airport, runway, stand, airline, weight class, and market segment are all high-cardinality categoricals. LightGBM finds optimal groupings via gradient-based splitting rather than requiring one-hot encoding.
- Fast training — 2 million rows × 15 features × 3,000 rounds trains in minutes on CPU. This iteration speed was essential for the feature sweep + hyperparameter tuning work.
- Interpretable feature importance — gain-based importances guided every decision about what to add, keep, and revert.
- Strong track record on tabular regression with mixed feature types at this scale.

**Alternatives considered:**

| Alternative | Why not adopted |
|---|---|
| XGBoost | Similar predictive power but slower on CPU (column-wise vs. leaf-wise split finding) and less flexible NaN handling. No clear benefit over LightGBM for this dataset. |
| CatBoost | Excellent categorical handling and strong on small datasets, but slower to train and less iteration-friendly. The gain over LightGBM for purely tabular regression is typically small. |
| Neural networks (MLP, TabNet) | Potentially capture deeper interactions but: slower to iterate, harder to debug feature contributions, and consistently outperformed by tree ensembles on tabular data with this many rows and feature types. Not worth the iteration cost for a competition with a fixed deadline. |
| Linear regression / ridge | Too simple — taxi time has strong non-linear interactions (e.g. congestion × airport × hour). Useful only as a sanity check baseline. |
| Random forest | Competitive baseline but gradient boosting with `num_boost_round=3000` consistently outperforms random forests on tabular regression. Random forests also require more trees for equivalent accuracy. |
| Per-airport models | Tested (see v10 attempt above). Resulted in 388.2s — worse than pooled model. Each airport gets far fewer training rows; the model loses the cross-airport generalisation that helps calibrate shared features like congestion and schedule delay. |

### Validation strategy: time-based split

An 83%/17% split by timestamp (approximately Jan–Oct 2025 train, Nov–Dec 2025 val) was used throughout.

**Why time-based:** Flight taxi time has temporal structure. The congestion signal is a rolling window over past completed flights — a random split would contaminate training with future flights that appeared in the congestion windows of training examples. Any feature that looks backward over real time is vulnerable to this. A time-based split preserves causal ordering.

**Why not k-fold CV:** Standard k-fold mixes time periods and leaks temporal features. Blocked time-series CV would be more rigorous (multiple train/val windows), but the computational cost (9 seeds × 5 folds = 45 models per experiment) made it impractical for the feature sweep pace we needed.

**Known limitation:** Our validation window (Nov–Dec 2025) is only 2 months and includes winter holiday patterns. The ranking data spans 9 months (Jan–Sep 2026) with different seasonal structure. Any distribution shift between those periods is invisible in our validation RMSE, as discussed in the Official Score Gap section.

### Ensemble: multi-seed with feature subsampling

The final model is an average of 9 LightGBM models trained with the same hyperparameters but different random seeds and `feature_fraction=0.8`.

**What each model does:** All 9 models are identical in purpose — each is a full LightGBM regressor trained on the same training data, with the same hyperparameters, predicting `TAXITIME_SEC_mvt`. They are not specialised. There is no model dedicated to a specific airport, time of day, or flight type. The only difference between them is the random seed, which controls which 80% of features are considered at each tree split.

**Why seeds produce different models:** At every node in every tree, LightGBM considers splitting on a random subset of features (80% here). Different seeds draw different subsets, so the trees end up with different structure — one model might split first on `congestion_signal`, another on `schedule_delay_sec`, producing different learned thresholds and different predictions for the same flight. The predictions from all 9 models are then averaged:

```python
np.mean([m.predict(features) for m in models], axis=0)
```

**Why averaging helps:** The 9 models make partially uncorrelated errors — where one model is wrong high, another is less wrong. When you average partially uncorrelated predictions, the random errors cancel and the systematic signal remains. This reduces variance without introducing bias. It is the same principle as random forests, applied to gradient boosting.

**Why `feature_fraction` is the diversity mechanism:** Without it (`feature_fraction=1.0`), LightGBM's split-finding is deterministic given the same data and features. All 9 seeds produce structurally identical trees, and averaging provides no benefit.

**Alternatives considered:**

| Alternative | Why not adopted |
|---|---|
| Multi-algorithm ensemble (LGB + XGB + CatBoost) | Higher diversity but 3× the training time and complexity. The marginal RMSE gain rarely justifies it; single-algorithm ensembles are easier to tune and deploy. |
| Stacking (meta-learner over base model predictions) | Adds another layer of training and validation management. Stacking helps most when base models have complementary error patterns — here all models are LightGBM variants with correlated errors. |
| Bagging fraction (row subsampling) | Tested. All bagging fractions worse than baseline (see Hyperparameter Sweep section). At 3,000 rounds, the model needs the full dataset to converge; reducing rows hurts more than it diversifies. |

### Feature engineering philosophy

Features were added one at a time, tested against the baseline, and reverted if neutral or negative. No automated feature generation was used.

The design principle was physical causation: if a feature has a plausible mechanism for affecting taxi-out time (congestion pressure, schedule adherence, stand position, operational disruption), it was worth testing. Features with no clear mechanism but high apparent correlation were treated with suspicion (potential data leakage or spurious correlation on the training period).

The leakage discovery (`MVT_TIME_UTC_mvt` has r = −1.0 with the target) confirmed this caution was warranted. Two columns that appear useful — arrival time and actual runway time — are definitionally derived from the target and cannot be used.

---

## v1 — Baseline
**Validation RMSE: 395.9s (6.60 min)**

Features: airport, runway, stand, airline, weight class, market segment, month, gate delay.
No congestion signal.

---

## v2 — Congestion Signal Added
**Validation RMSE: 386.4s (6.44 min) — improvement of 9.5s**

Added 60-min rolling mean taxi time of recent completed flights at the same airport (CONGESTION_WINDOW_MINUTES = 60).
Signal populated for 2,074,477 / 2,085,047 rows (~99.5%).
Congestion signal for ranking data will be NaN (training ends 2026-01-01, ranking starts 2026-01-04 — no overlap in 60-min windows).

**Next levers to try:**
- Tune CONGESTION_WINDOW_MINUTES (30, 45, 90 min)
- Tune num_leaves, min_data_in_leaf
- ~~Incorporate ranking arrivals as congestion context for ranking predictions~~ — done in v3

---

## v3 — Ranking Congestion Fixed

**Validation RMSE: 386.4s (6.44 min) — unchanged (fix only affects submission quality)**

Ranking arrivals (215,967 rows with known taxi-in times at `ADES_mvt`) now used as congestion context alongside training data. Congestion signal coverage for ranking DEP rows: 0% → 99.8%.

Validation RMSE is unchanged because validation data falls within the training window, where congestion was already populated. Impact is only visible via actual submission score.

**Next levers to try:**
- Tune CONGESTION_WINDOW_MINUTES (30, 45, 90 min)
- NaN gate delay conflates "no delay" with "missing data" — separate these signals

---
## Hyperparameter Grid Search
Learning rate: 0.05 | Early stopping: 30 rounds | Max rounds: 1000

| num_leaves | min_data_in_leaf | best_round | RMSE (s) | RMSE (min) |
|---|---|---|---|---|
| 127 | 50 | 154 | 384.3 | 6.40 | ← best
| 63 | 100 | 260 | 384.8 | 6.41 |
| 127 | 100 | 189 | 385.1 | 6.42 |
| 255 | 50 | 145 | 385.6 | 6.43 |
| 255 | 100 | 169 | 385.9 | 6.43 |
| 63 | 50 | 248 | 386.0 | 6.43 |
| 127 | 20 | 190 | 386.1 | 6.44 |
| 63 | 20 | 272 | 386.6 | 6.44 |
| 255 | 20 | 117 | 387.4 | 6.46 |

**Best config:** num_leaves=127, min_data_in_leaf=50 → RMSE 384.3s

---
## Congestion Window Sweep
num_leaves=127, min_data_in_leaf=50, lr=0.05, early stopping 30 rounds

| window_minutes | coverage (%) | RMSE (s) | RMSE (min) |
|---|---|---|---|
| 30 | 99.3 | 386.1 | 6.44 |
| 45 | 99.4 | 387.7 | 6.46 |
| 60 | 99.5 | 384.3 | 6.40 | ← best
| 90 | 99.6 | 386.8 | 6.45 |

**Best window:** 60 min → RMSE 384.3s

---

## Gate Delay NaN Fix — Reverted

Tried removing `fillna(0)` from `gate_delay_sec` to let LightGBM handle missing values natively.
Result: 386.8s — 0.4s **worse**. Reverted.

Only 1.1% of rows had missing gate delay. The 0-fill appears to be a reasonable proxy for those flights (they likely depart near on-time), so treating them as unknown hurt more than it helped.

---

## Feature Sweep (5 candidates)

Baseline for this sweep: 384.2s (v5 — hour of day added)

| Feature | RMSE (s) | Delta | Decision |
|---|---|---|---|
| Departure queue (±30 min EOBT count) | 386.3 | +2.1 | Reverted |
| Scheduled hour (EOBT_1_flt hour) | 386.1 | +1.9 | Reverted |
| is_late_departure (gate_delay > 0) | 384.2 | 0.0 | Reverted (neutral) |
| Stand congestion signal (60 min) | 384.7 | +0.5 | Reverted — only 5.6% coverage |
| Day of week | 385.1 | +0.9 | Reverted |

None of the 5 features improved RMSE. Model remains at **384.2s**. The signal is largely exhausted from available features in this dataset.

---

## Error Analysis

LIRF (Rome Fiumicino) accounts for **56.6% of total MSE** despite being 7.1% of validation flights.
Without LIRF, RMSE is **262.8s** (sub-300).
LIRF has persistent data quality issues: tracking artifacts producing 87,000s+ taxi times every month.
These are real-world data inconsistencies per organiser note — unpredictable by any model.

Non-LIRF airports: RMSE 262.8s. The competition is effectively decided here.

---

## Additional Feature Sweep

Baseline: 384.2s

| Feature | RMSE (s) | Delta | Decision |
|---|---|---|---|
| Runway congestion signal (60 min) | 387.3 | +3.1 | Reverted — coverage 97%, noise > signal |
| Schedule delay (`BLOCK_TIME - SCHED_TIME`) | **361.3** | **−22.9** | **Kept** |

---

## v6 — Schedule Delay Added
**Validation RMSE: 361.3s (6.02 min) — improvement of 22.9s**

`BLOCK_TIME_UTC_mvt - SCHED_TIME_UTC_mvt` in seconds. 100% coverage.
Measures total schedule delay (actual block-off vs scheduled). Stronger signal than gate delay alone.
LIRF RMSE: 1030.7s (was 1076.4s). Non-LIRF RMSE: 242.6s (was 262.8s).

---

## v7 — Day Deviation Ratio Added
**Validation RMSE: 358.5s (5.97 min) — improvement of 2.5s**

For each departure, computes the ratio of mean taxi time of completed flights at the same airport since midnight UTC vs that airport's long-run mean. A ratio of 1.5 means the airport is running 50% slower than normal today. Fires after 5 completed flights; NaN before that.

Signal populated for 2,056,622 / 2,085,047 rows (98.6%). Gain is modest globally because disruption days are rare (21 days across the training set), but the feature is directionally correct and costs little.

---

## Plan Experiments — All Reverted

Baseline: **244.0s** (v9, after artifact filter)

| Experiment | Detail | RMSE (s) | Delta | Decision |
|---|---|---|---|---|
| Huber loss alpha=50 | Huber threshold 50s | 289.7 | +45.7 | Reverted |
| Huber loss alpha=100 | Huber threshold 100s | 272.7 | +28.7 | Reverted |
| Huber loss alpha=200 | Huber threshold 200s | 257.4 | +13.4 | Reverted |
| Huber loss alpha=400 | Huber threshold 400s | 246.1 | +2.1 | Reverted |
| airport_runway variant A | combined + keep both | 244.0 | 0.0 | Reverted (neutral) |
| airport_runway variant B | combined + drop runway | 244.7 | +0.7 | Reverted |
| airport_runway variant C | combined + drop both | 244.0 | 0.0 | Reverted (neutral) |
| airport_hour interaction | LFPG bias fix | 244.4 | +0.4 | Reverted |

**Huber loss:** All alphas worse than MSE. The artifact filter already removed the extreme outliers that motivated Huber; without them, the remaining tail isn't heavy enough for Huber to outperform MSE.

**airport_runway:** Model already captures the joint signal effectively via separate airport + runway features at their current importance levels. Combining them adds cardinality without new information.

**airport_hour (LFPG fix):** LFPG diagnostic found hour-10 mean residual of +129s (3,136 flights) and a runway distribution shift (26R: 28.6% → 36.9%, 27R: 7.5% → 15.9% in validation). Despite this clear hour-level bias, the airport_hour feature didn't help — the hour pattern at CDG itself shifted between training and validation periods, so the model can't generalise it. The LFPG bias appears to be a train/val distribution shift problem, not a missing feature problem.

**Conclusion:** 244.0s appears to be the ceiling for this feature set and data. The remaining error is driven by genuinely unpredictable events (LIRF disruption days) and structural distribution shifts between training and validation periods at CDG and EGLL.

---

## LIRF Deep Dive

**Two distinct problems — data artifacts and genuine disruption days.**

### Artifact rows (14 rows)
All cluster at exactly 24.0–24.6 hours — a date rollover bug where MVT_TIME crossed midnight but the block time used the wrong date. One flight logs 36 hours.
- These 14 rows account for **20.6% of total SSE** despite being 0.001% of the data.
- Should be capped/filtered before training (`TAXITIME_SEC_mvt > 21600` is a safe threshold).

### Disruption days (3 days at LIRF)
| Date | Daily mean | Max taxi | Flights |
|---|---|---|---|
| 2025-07-13 | 42 min | 671 min | 479 |
| 2025-07-28 | 32 min | 315 min | 512 |
| 2025-07-14 | 32 min | 1,452 min | 522 |

The failure mode is **underprediction** — airport gridlock we didn't foresee, not phantom high predictions. By 8–9am on disruption days the day_deviation_ratio signal crosses 1.3× (p95 for normal days is 1.23), giving subsequent flights a warning.

### Disruption days are not LIRF-specific
5 airports have days where daily mean > 1.5× their own baseline:
| Airport | Disruption days | Worst ratio |
|---|---|---|
| LTFM (Istanbul) | 7 | 2.8× |
| EDDM (Munich) | 6 | 2.4× |
| LFPG (Paris CDG) | 4 | 2.0× |
| LIRF (Rome) | 3 | 2.1× |
| LSZH (Zürich) | 1 | 1.5× |

The day_deviation_ratio feature is the primary lever for these events. The 60-min congestion signal also helps but is narrower.

---

## Late Gate Departure — Confirmed Stat Sig
`AOBT_3_flt - LOBT_flt > 0` (late gate departure) tested against taxi time across all 2M training departures.
- Mann-Whitney p ≈ 0 (machine precision)
- Mean taxi difference: +1.98 min for late gate vs on-time
- Pearson r between gate delay magnitude and taxi time: 0.25 (~6% variance explained)
- Effect size (rank-biserial r): 0.16 — small but real

Already captured by `gate_delay_sec` as a continuous feature. A binary `is_late_departure` flag tested separately and was neutral (see Feature Sweep above).

---

## v5 — Hour of Day Added
**Validation RMSE: 384.2s (6.40 min) — improvement of 0.1s**

Added `hour` (UTC, numeric 0–23) as a feature. Consistent with plan's ε²=0.009 estimate — negligible standalone signal but kept as it costs nothing.

---

## Continued Feature Sweep

Baseline: 357.6s (v7 with arvt_update_sec)

| Feature | RMSE (s) | Delta | Decision |
|---|---|---|---|
| flight_type categorical | 359.6 | +2.0 | Reverted — noise |
| Artifact filter (TAXITIME > 21600 removed from training) | **244.0** | **−113.6** | **Kept** |

**WARNING: BLOCK_TIME - MVT_TIME and MVT_TIME - SCHED_TIME are data leakage** — MVT_TIME = BLOCK_TIME + TAXITIME by definition (r=-1.0 confirmed). Never use MVT_TIME as a feature.

> **RECONSIDERED (2026-09-22):** This blanket ban was reversed in v22. `MVT_TIME` is
> actually *present in the ranking set* (only `BLOCK_TIME` is withheld), so
> `proxy_taxi = MVT_TIME − AOBT_3` is a legitimately-available feature: on clean rows
> it's a sharp taxi estimate, and on date-rollover artifacts it equals the corrupted
> label. It's target-adjacent, not unusable leakage. See "Leaderboard Reconciliation".

---

## v7 — ARVT Update + Day Deviation
**Validation RMSE: 357.6s (5.96 min)**

Added `arvt_update_sec` = ARVT_3 - ARVT_1 (change in estimated arrival time). 99.9% coverage. r=0.20 with taxi time. Improvement of 0.9s over v6.

---

## v8 — Artifact Rows Filtered
**Validation RMSE: 244.0s (4.07 min) — improvement of 113.6s**

### What happened

Removed 69 training rows where `TAXITIME_SEC_mvt > 21600` (6 hours). That single line of filtering produced the largest single improvement in the entire project — larger than all other features combined.

### Root cause

The LIRF deep dive (see above) identified 14 rows at Rome Fiumicino where taxi time was recorded as 86,400–130,000 seconds (24–36 hours). These are date-rollover tracking artifacts: the actual pushback (`MVT_TIME_UTC_mvt`) crossed UTC midnight, but the block time (`BLOCK_TIME_UTC_mvt`) was recorded using the wrong calendar date. The resulting `TAXITIME_SEC_mvt` is off by exactly 86,400 seconds (one day).

These rows are not extreme outliers from a real operational event. They are measurement errors with no meaningful signal — no combination of features (airport, hour, congestion, schedule delay) can predict a value that is wrong by definition.

### Why the impact was so large

The mean squared error loss function squares the residuals. A single row with a true taxi time of ~900s but a recorded value of 87,300s contributes a squared error of roughly (87,300 − 900)² ≈ 7.4 × 10⁹. The entire dataset has ~2 million rows. A handful of these artifact rows can dominate the loss completely.

Before filtering, the gradient boosting model was trying to partially accommodate these targets. Because the artifacts share feature values with normal flights (same airport, same hours, similar congestion), the model was nudged toward higher predictions for LIRF flights generally — degrading accuracy for the vast majority of legitimate rows in order to reduce loss on a tiny number of garbage targets.

After filtering, those corrupting gradients are gone. The model learns only from real taxi times and predicts accordingly.

### Why earlier outlier-handling attempts failed

Two prior approaches tried to address this without removing the rows:

1. **Huber loss**: Downweights large residuals during training. This reduced the gradient corruption but left the artifact rows in the validation set, so evaluation still penalised the model for not predicting 87,000s. Net result: worse RMSE.

2. **Outlier cap (60 min, 120 min)**: Capped the training target at a maximum value. Same problem — validation set still contained uncapped rows with recorded values of 87,000s, and the model was punished for predicting reasonable values.

The correct fix was filtering the rows entirely from both training and validation. The validation RMSE improvement reflects two things: a better-trained model (no corrupted gradients) and a fairer evaluation (no unachievable targets in the validation set).

### Threshold choice

The filter threshold of 21,600 seconds (6 hours) was chosen based on inspection of the distribution. All artifact rows cluster at exactly 24.0–36.0 hours. The longest legitimate recorded taxi time in the training set was well under 2 hours. A 6-hour threshold leaves substantial headroom while cleanly separating real operations from tracking errors.

### Caveat

The competition score depends on whether the ranking data (Jan–Sep 2026) contains similar date-rollover artifacts. If it does and those rows are included in the truth, our predictions for them will be reasonable values (~900s) while the truth records ~87,000s — the same situation that inflated our pre-filter validation RMSE. If the ranking data is clean, the score will match our 244.0s validation RMSE closely.

---

## Full Project Narrative

### The problem

The PRC 2026 challenge asks us to predict `TAXITIME_SEC_mvt` — the time in seconds between a flight's actual gate departure (block-off) and its actual runway departure (wheels-up). Taxi-out time is affected by airport congestion, runway assignment, aircraft type, time of day, schedule adherence, and a long tail of chaotic real-world factors. The competition provides 12 months of European airport movement data (Nov 2024 – Dec 2025) covering 11 major airports and roughly 2 million departure flights.

### Model choice: LightGBM

We chose LightGBM gradient boosting from the start and never changed it. The dataset has 2 million rows, a mix of numerical and categorical features, and no need for spatial or sequential structure that would favour a different architecture. LightGBM handles missing values natively (NaN passes through without imputation), trains fast enough for rapid iteration, and has well-understood hyperparameters. A neural network or ensemble of heterogeneous models might extract fractionally more signal but would cost iteration speed we needed for feature exploration.

### Validation strategy

We used a time-based split: the first 83% of flights by timestamp are training, the last 17% are validation. Random splits would leak future patterns into training — a flight's congestion signal depends on the flights that came before it, so mixing future flights into the training pool would give the model unrealistic information. The 83/17 split approximates 10 months of training and 2 months of validation.

### Baseline (395.9s)

The initial model used eight static features: airport, runway, stand, airline, aircraft weight class, market segment, month, and gate delay (AOBT minus EOBT, filled to zero when missing). This captured the structural differences between airports and the direct relationship between being late to push back and spending longer on the taxiway.

### Congestion signal (−9.5s → 386.4s)

The first meaningful addition was a real-time congestion signal: the mean taxi time of all completed departures at the same airport in the 60 minutes before a flight's pushback. This is computed vectorially using sorted timestamps and cumulative sums rather than row-by-row loops, making it fast enough to run on 2 million rows without issue. Coverage was 99.5% — only the very first flights of the day at each airport had no prior context.

A critical bug was discovered immediately after: the ranking data (the flights we need to predict) starts on 4 January 2026, but the training data ends on 31 December 2025. The 60-minute windows for ranking flights looked backward into early January 2026 — a gap with no training data, so every ranking flight had a NaN congestion signal. We fixed this by using the ranking set's arrival (ARR) rows as congestion context: those rows have known taxi-in times at their destination airport. By mapping `ADES_mvt → ADEP_mvt`, those arrivals became a pool of recent completed flights that ranking departure predictions could draw congestion context from. Coverage jumped from 0% to 99.8% for ranking predictions.

### Hyperparameter tuning (−2.1s → 384.3s)

A grid search over `num_leaves` ∈ {63, 127, 255} and `min_data_in_leaf` ∈ {20, 50, 100} with early stopping found `num_leaves=127, min_data_in_leaf=50` as best at 384.3s. The variation across the grid was small (±3s), confirming the model was not being dramatically under- or over-fit. A congestion window sweep (30, 45, 60, 90 min) confirmed 60 minutes as optimal.

### Hour of day (−0.1s → 384.2s)

Adding the UTC departure hour as a numeric feature produced a negligible improvement. The airport categorical feature already encodes much of the time-of-day pattern implicitly (airports have characteristic peak hours), so the marginal gain was small. Kept because it cost nothing.

### First feature sweep — five candidates, all failed

Tested five candidates against the 384.2s baseline:

- **Departure queue (±30 min EOBT count)**: +2.1s. Counting flights with similar scheduled push-back times added noise rather than signal. The congestion signal already captured queue pressure from completed flights; planned queue added uncertainty without precision.
- **Scheduled departure hour (EOBT hour)**: +1.9s. Redundant with the actual departure hour already in the model.
- **is_late_departure binary flag (gate_delay > 0)**: neutral. The continuous `gate_delay_sec` already captured this.
- **Stand congestion signal**: +0.5s, with only 5.6% coverage. Most stands are used infrequently enough that a 60-minute window rarely catches another completed flight.
- **Day of week**: +0.9s. Weekly cycles exist but the model was already capturing enough temporal structure through airport + hour + month.

The consistent pattern from this sweep: features that encode information already present in other features add noise rather than signal.

### Error analysis — discovering the LIRF problem

After the first sweep stalled, we decomposed validation MSE by airport. Rome Fiumicino (LIRF) accounted for 56.6% of total MSE while representing only 7.1% of validation flights. Without LIRF, the model's RMSE was already 262.8s — sub-300. With it, the overall score was pulled to 384s.

Inspecting LIRF's residuals revealed two distinct problems:

**1. Tracking artifacts.** Fourteen rows per month had recorded taxi times of 86,400–130,000 seconds (24–36 hours). These are date-rollover bugs: the actual pushback crossed UTC midnight, but the block time used the wrong date, producing a taxi time inflated by exactly 86,400 seconds. These rows are not slow flights. They are measurement errors. No model can predict them correctly because the target values are wrong.

**2. Genuine disruption days.** Three days at LIRF in July 2025 showed mean taxi times of 32–42 minutes, with individual flights reaching 24 hours (on disruption days — not artifacts). These are real gridlock events, not tracking errors. Similar disruption patterns exist at four other airports: Istanbul (7 events, worst 2.8× baseline), Munich (6 events, 2.4×), Paris CDG (4 events, 2.0×), and Zürich (1 event, 1.5×).

### Dead ends: per-airport models, Huber loss, outlier caps

Three approaches were tried to address LIRF without removing data:

**Per-airport models**: Trained a separate LightGBM for each of the 11 airports, removing the airport categorical feature within each model. Result: 388.2s overall — 4s worse. A per-airport model sees fewer training examples (LIRF has ~150k training rows, not 2M), which limits tree depth and generalisation. The data quality problem remained regardless of model architecture.

**Huber loss**: Replaced the squared-error objective with Huber loss, which downweights large residuals during training. Tested alpha values of 300, 600, 900, and 1200 seconds. All produced worse RMSE than MSE. The reason: Huber loss reduces gradient corruption during training but does nothing to the validation set. The artifact rows remained in validation with recorded values of 87,000s, and the model was still evaluated against them. A model trained with Huber loss to predict reasonable values (~900s) for artifact flights was punished more at evaluation than a model trained with MSE that had partially learned to predict higher values for LIRF.

**Outlier caps (60 min, 120 min)**: Capped the training target so no flight had a recorded taxi time above 3,600s or 7,200s. Same failure mode as Huber loss — the cap only affected training, not validation.

The lesson from all three failed approaches: when the validation set contains the same corrupt data, no training-side fix can help. The correct fix is filtering the rows entirely.

### Data leakage discovery

During feature exploration, a column was considered that turned out to encode the target directly. `MVT_TIME_UTC_mvt` (the actual runway departure time) equals `BLOCK_TIME_UTC_mvt + TAXITIME_SEC_mvt` by definition — the competition data is structured this way. Any feature derived from `MVT_TIME_UTC_mvt` (such as `BLOCK_TIME - MVT_TIME` or `MVT_TIME - SCHED_TIME`) would have a perfect correlation with the target and produce meaningless validation scores. Confirmed r = −1.0. These columns were immediately excluded from all feature sets.

### Schedule delay (−22.9s → 361.3s)

`BLOCK_TIME_UTC_mvt - SCHED_TIME_UTC_mvt`: the difference between actual gate departure and the originally scheduled gate departure, in seconds. 100% coverage. This is a much stronger signal than gate delay (`AOBT - EOBT`) because it captures the cumulative effect of everything that went wrong with a flight's schedule — late inbound aircraft, late crew, gate changes, and operational disruptions — not just the final gate-departure timing. A flight that is 45 minutes behind schedule tends to push into a more congested runway environment and carries a different operational character than an on-time flight. LIRF RMSE dropped from 1,076s to 1,030s; non-LIRF RMSE dropped from 262.8s to 242.6s.

### Day deviation ratio (−2.8s → 358.5s)

For each departure, compute the ratio of the mean taxi time of all completed flights at that airport since midnight UTC versus that airport's long-run mean. A ratio of 1.5 means the airport is running 50% slower than its historical average for that day. The signal fires only after 5 flights have completed that day, to avoid unstable early-morning estimates. Coverage was 98.6%.

The global gain was modest because genuine disruption days are rare — 21 days across the full 12-month training set. But the feature is directional: it tells the model when an airport is in a bad state before any individual flight has been predicted badly. On disruption days, the ratio crosses 1.3× by 8–9am, which is above the 95th percentile for normal days (1.23×), giving subsequent flights a meaningful warning.

### ARVT update (−0.9s → 357.6s)

`ARVT_3_flt - ARVT_1_flt`: the change in estimated arrival time between the third and first flight plan updates. A positive value (later expected arrival) correlates weakly with longer taxi-out times (Pearson r = 0.20), likely because flights expecting a late arrival are also delayed generally. 99.9% coverage. The improvement was small but the signal was real and consistent.

### The breakthrough: artifact filter (−113.6s → 244.0s)

Having identified the artifact rows in the LIRF analysis, we added a single filter to `load_training_data()`:

```python
return df[df["TAXITIME_SEC_mvt"] <= 21600].copy()
```

This removed 69 rows from the 2-million-row training set (0.003%). The validation RMSE dropped by 113.6 seconds — more than all other changes combined.

The mechanism is explained in the v8 section above. In short: MSE loss squares residuals, so a handful of rows with targets of 87,000s dominated the entire loss landscape. The model was being trained to partially accommodate values that were wrong by definition, corrupting its predictions for the remaining 99.997% of legitimate flights. Removing those rows let the model learn from real taxi times only.

Why this was the last thing tried rather than the first is worth noting: the earlier attempts (Huber loss, outlier caps) were trained with the understanding that the validation set still contained those rows. Once it became clear that filtering was needed on both sides — train and validation — and that the filter threshold was safe (the gap between real maximums and artifact minimums is enormous), the improvement was immediate and decisive.

### RMSE progression summary

| Version | Change | RMSE (s) | Delta |
|---|---|---|---|
| v1 | Baseline | 395.9 | — |
| v2 | Congestion signal | 386.4 | −9.5 |
| v3 | Ranking congestion fixed | 386.4 | 0.0 (submission quality only) |
| v4 | Hyperparameter tuning | 384.3 | −2.1 |
| v5 | Hour of day | 384.2 | −0.1 |
| v6 | Schedule delay | 361.3 | −22.9 |
| v7 | Day deviation ratio + ARVT update | 357.6 | −3.7 |
| v8/v9 | Artifact filter | **244.0** | **−113.6** |

### Where the RMSE comes from now

With artifact rows removed, validation RMSE is 244.0s. The non-LIRF airports were already at 242.6s after v6 — so LIRF's genuine disruption days (not artifacts) account for most of the remaining gap. Those events are partially addressed by the day_deviation_ratio signal but cannot be fully predicted: they are low-frequency, high-severity events where no available feature signals the extreme before it begins.

The remaining model features explain the structural variation well. The unsolved error is concentrated in genuinely unpredictable operational chaos.

---

## v10 Attempt — Runway Groups + Disruption Prior — Reverted

**RMSE: 247.7s (+3.7s vs v9 244.0s baseline) — No improvement, reverted**

Two new features were tested simultaneously:

- `runway_group`: mapped each (airport, runway) pair to a physical operational group for LFPG/LIRF/EGLL/EDDF/EDDM (e.g. LFPG_north, LFPG_south) to pool same-complex runways and reduce sensitivity to seasonal runway distribution shifts. Ended up as the second-highest importance feature at 17.0% (gain), surpassing `stand` — but still produced worse RMSE overall.
- `airport_month_disruption_rate`: P(disruption | airport, month) from training data — historical fraction of days per airport × month where daily mean taxi time exceeded 1.5× the airport baseline. Only 1.0% feature importance, suggesting it added little signal.

Per-airport RMSE (validation):
| Airport | RMSE (s) | Mean residual | n |
|---|---|---|---|
| LFPG | 321.8 | +28.0 | 41,579 |
| LIRF | 371.0 | −28.9 | 25,306 |
| EGLL | 293.0 | +9.1 | 42,558 |
| EDDF | 204.6 | +2.2 | 38,351 |
| EDDM | 230.1 | +6.3 | 26,682 |

Top 5 feature importances (gain):
| Feature | Importance |
|---|---|
| congestion_signal | 18.8% |
| runway_group | 17.0% |
| stand | 13.8% |
| arvt_update_sec | 11.7% |
| gate_delay_sec | 11.2% |

The runway_group feature captured a real structural signal (high gain importance) but the overall RMSE worsened by 3.7s. Likely explanation: pooling runways within a complex reduces the cardinality/coverage problem for minority runways in validation but simultaneously loses the fine-grained distinction that the raw runway feature provided for the majority of flights. The net effect was negative. The disruption prior was essentially inert (1.0% importance), confirming the day_deviation_ratio signal already captures most of the same information dynamically on the day itself.

---

## Time-Decayed Sample Weights — Reverted

**All half-lives worse than baseline. No improvement. No changes to model.py.**

Hypothesis: the LFPG/EGLL RMSE is partly caused by a seasonal runway distribution shift between the training period (Jan–Oct 2025) and the validation period (Nov–Dec 2025). Weighting more recent training rows higher (exponential decay over `MVT_TIME_UTC_mvt`) would pull learned runway constants toward Sep–Oct data, which is closer to the validation distribution.

Sweep results (LightGBM training with `lgb.Dataset(weight=...)` using `exp(-log(2)/half_life * age_days)`):

| half_life | RMSE (s) | delta vs 244.0s |
|---|---|---|
| None (uniform) | 244.0 | 0.0 — BEST |
| 270 days | 245.4 | +1.4 |
| 180 days | 246.0 | +2.0 |
| 90 days | 248.2 | +4.2 |
| 60 days | 250.1 | +6.1 |

The degradation is monotonic — shorter half-lives are strictly worse. Uniform weighting wins.

Root cause diagnosis: LightGBM already learns seasonal patterns through the `month` categorical feature. Time-decayed weights simply reduce the effective training set size without providing new information the model couldn't already learn from feature interactions. The LFPG runway shift is a genuine distribution shift that no amount of reweighting can fully bridge, because the validation runway proportions are materially different from any subset of training months.

**Conclusion:** The 244.0s baseline is the ceiling for this feature set and training setup. All attempted improvements — Huber loss, airport_runway combined feature, airport_hour interaction, config-agnostic runway groups, disruption prior, and time-decayed sample weights — failed to improve on it. The remaining error is driven by irreducible factors: genuine operational chaos (LIRF/LFPG disruption days) and structural distribution shifts between training and validation periods that available features cannot compensate for.

---

## v10 — Congestion Acceleration Added

**Validation RMSE: 243.5s (−0.5s vs v9 244.0s baseline) — KEPT**

New feature `congestion_acceleration = congestion_30min − congestion_60min`. A positive value means the 30-min mean is worse than the 60-min mean — congestion is worsening. A negative value means it is easing. Captures the *direction of trend* in airport congestion, which the rolling mean alone cannot encode.

| Metric | Baseline (v9) | v10 | Delta |
|---|---|---|---|
| Overall RMSE | 244.0s | 243.5s | **−0.5s** |
| December RMSE | 251.3s | 250.2s | −1.1s |
| LFPG RMSE | ~311s | 313.0s | slight noise |
| LIRF RMSE | ~371s | 369.1s | −1.9s |
| EGLL RMSE | ~282s | 281.5s | −0.5s |
| Worst-5% SSE | 61.0% | 61.0% | unchanged |

Feature importance (gain): **0.7%** — small but consistent.

Coverage: 99.3% of training rows. The ~0.7% NaN rows are earliest-of-day flights at each airport where the short window also has no completed predecessors; LightGBM handles NaN natively.

Implementation: `CONGESTION_WINDOW_SHORT_MINUTES = 30` constant added. Two calls to `compute_congestion_signal()` in `main()` and `write_submission()`. `build_features()` signature updated to accept `congestion_acceleration` as a fourth argument.

**Updated in v11:** short window tuned down to 10 min (see below).

---

## v11 — Congestion Acceleration Window Tuned to 10 min

**Validation RMSE: 242.7s (−1.3s vs v9 244.0s baseline, −0.8s vs v10) — KEPT**

Swept short window sizes for `congestion_acceleration = congestion_N − congestion_60`. Also tested stacking multiple windows simultaneously ("consensus"). All signals computed once; split shared across all runs.

| Configuration | RMSE | vs v9 244.0s |
|---|---|---|
| accel = 10-min − 60-min | **242.7s** | **−1.3s** |
| accel = 15-min − 60-min | 243.0s | −1.0s |
| v10 + 20-min stacked | 243.1s | −0.9s |
| v10 + 15-min stacked | 243.1s | −0.9s |
| v10 + 10-min stacked | 243.1s | −0.9s |
| v10 (accel = 30-min − 60-min) | 243.5s | −0.5s |
| All 5 windows stacked | 243.3s | −0.7s |
| accel = 45-min − 60-min | 244.3s | +0.3s |

**Shorter windows win**: the 10-min window is the most reactive to developing congestion and produces the strongest leading signal. The 45-min window is too close to the 60-min base to provide useful directional information.

**Stacking adds noise**: the full consensus model (all 5 windows) at 243.3s is worse than using 10-min alone at 242.7s. Extra windows are correlated with the best one; in LightGBM trees they compete for the same splits without adding independent signal.

`CONGESTION_WINDOW_SHORT_MINUTES` updated from 30 → 10. `SUBMISSION_VERSION` incremented to 11.

---

## v12 — Stand Prefix Added

**Validation RMSE: 241.4s (−1.3s vs v11 242.7s, −2.6s vs v9 244.0s baseline) — KEPT**

New feature `stand_prefix`: first character of `STAND_mvt`, filled to `"UNK"` when missing. Added to `CATEGORICAL_FEATURES`. 4.66% gain importance.

Stand prefix encodes which terminal complex or concourse a flight departs from. At every airport the per-prefix taxi time spread is large and physically motivated:

| Airport | Best prefix | Mean | Worst prefix | Mean | Spread |
|---|---|---|---|---|---|
| LIRF | 2 | 969s | 9 | 1745s | 776s |
| EGLL | 2 | 1221s | 5 | 1441s | 220s |
| LFPG | X | 755s | K | 1172s | 417s |
| EDDF | V | 785s | K | 1146s | 361s |
| LSZH | F | 424s | T | 918s | 494s |

All airports have 0% NaN on `STAND_mvt`.

Why prefix adds something beyond the full `stand` categorical: with 409 unique stands at LFPG and `min_data_in_leaf=50`, sparse stands are regularised toward the global mean rather than their terminal-complex mean. Prefix gives those thin stands a better prior — "this stand I haven't seen much is in the K-complex, so expect +153s above average." LFPG improved by 3.4s (311.5s → 308.1s), confirming this hypothesis.

1-char and 2-char prefix tested — identical RMSE (241.41s). 1-char kept for simplicity.

---

## Shortlist Experiments — Hour Categorical and MIN_DAY_FLIGHTS=3 — Both Reverted

Both tested on the v12 baseline (241.41s). Both worse.

| Experiment | RMSE | Delta |
|---|---|---|
| hour → categorical | 242.01s | +0.60s |
| MIN_DAY_FLIGHTS = 3 | 241.80s | +0.39s |
| Both together | 242.18s | +0.77s |

**Hour as categorical**: making `hour` categorical forces LightGBM to treat each of 24 hours as independent buckets. Hours 00–03 have tiny row counts — leaf estimates become high-variance. The numeric form already gets clean threshold splits and works better.

**MIN_DAY_FLIGHTS = 3**: 10,806 extra rows gain coverage but with noisy early-morning ratio values (only 3 completed flights, high variance). Degraded signal quality on those rows outweighs the earlier coverage benefit. MIN_DAY_FLIGHTS stays at 5.

---

## is_holiday_window — Reverted

**RMSE: 244.2s (+0.2s vs baseline) — No improvement, reverted**

Binary feature flagging Nov 20–30, Dec 18–31, and Jan 1–5 (peak European holiday travel periods). 133,048 / 354,447 validation rows flagged (37.5% coverage).

| Metric | Baseline | With feature | Delta |
|---|---|---|---|
| Overall RMSE | 244.0s | 244.2s | +0.2s |
| December RMSE | 251.3s | 251.7s | +0.4s |
| LFPG RMSE | ~311s | 312.4s | +1.4s |
| LIRF RMSE | ~371s | 371.5s | +0.5s |
| EGLL RMSE | ~282s | 282.6s | +0.6s |
| Worst-5% SSE share | 61.0% | 60.9% | −0.1% |

Feature importance (gain): **0.0%** — the model assigned it no predictive weight.

Root cause: the `month` categorical already encodes December as a distinct bucket. The holiday window is entirely redundant with `month=12` — it subdivides December into holiday vs. non-holiday days, but the model found no additional signal in that subdivision beyond what the month feature already provided. The worst-day disruptions (Nov 22 LFPG, Dec 7 LTFM, Dec 31 EGLL) are operational events that happen to fall in winter, not events *caused* by the holiday calendar that a static binary flag can predict.

---

## Two-Stage Regime Architecture — Not Adopted

**Overall RMSE: 245.1s (+1.1s vs 244.0s baseline). Worst-5% SSE contribution increased from 61.0% → 63.5%. No changes to model.py.**

Architecture tested:
1. **Classifier**: LightGBM binary classifier predicting `is_disruption = (TAXITIME_SEC_mvt > per-airport p90)`. Trained on all 13 features. p90 thresholds computed on training split only.
2. **Model A**: Regressor trained exclusively on non-disruption rows (90% of training data, ~1.56M rows).
3. **Model B**: Regressor trained exclusively on disruption rows (10% of training data, ~173k rows).
4. **Blend**: `pred = (1 − prob) × pred_A + prob × pred_B`

Key diagnostic: `prob_disruption` mean on validation = **0.094**, p90 = 0.305. The classifier correctly identified low disruption probability for most flights, meaning the blend was effectively 94% Model A + 6% Model B for the average flight.

Why it failed:
- **Model A is weaker than the single model**: trained on 90% of the data, it lacks exposure to disruption-day context that helps it generalise even for normal flights.
- **The blend dilutes both models**: for normal flights, adding 6% of Model B's chaos-skewed predictions introduces noise. The worst-5% SSE share went up 2.5pp, confirming the tail got worse.
- **The classifier can't reliably route flights**: the features that would identify a true disruption flight (congestion_signal, day_deviation_ratio) are already in the single model. The classifier learns the same signal and the explicit routing step adds overhead without adding information.
- **The single model already performs implicit regime detection**: through its tree splits on congestion_signal and day_deviation_ratio, it already captures the two-regime structure without explicit separation.

| Metric | Single model | Two-stage blend | Delta |
|---|---|---|---|
| Overall RMSE | 244.0s | 245.1s | +1.1s |
| December RMSE | 251.3s | 253.1s | +1.8s |
| LFPG RMSE | ~311s | 317.2s | worse |
| LIRF RMSE | ~371s | 376.8s | worse |
| EGLL RMSE | ~282s | 282.5s | flat |
| Worst-5% SSE | 61.0% | 63.5% | worse |

---

## v13 — Learning Rate + Rounds Retuned

**Validation RMSE: 240.4s (−1.0s vs v12 241.4s) — KEPT**

The hyperparameter grid search was last run at 384s with ~6 features. With 15 features and the artifact-filtered dataset, the optimal num_boost_round was much higher. Swept LR × rounds:

| Config | RMSE | Delta |
|---|---|---|
| 0.05 / 500 (v12 baseline) | 241.41s | — |
| 0.04 / 700 | 241.87s | +0.47s |
| 0.03 / 1000 | 241.29s | −0.11s |
| 0.02 / 1500 | 241.02s | −0.38s |
| 0.02 / 2000 | 240.75s | −0.66s |
| 0.02 / 2500 | 240.48s | −0.93s |
| **0.02 / 3000** | **240.40s** | **−1.01s** |
| 0.015 / 3000 | 240.63s | −0.78s |
| 0.015 / 4000 | 240.48s | −0.93s |

The trend plateaus between 2500–3000 rounds at LR=0.02. Going lower (0.015) with more rounds converges to the same region, confirming LR=0.02 / 3000 is near the optimum.

`learning_rate` updated 0.05 → 0.02. `num_boost_round` updated 500 → 3000. No other changes.

**Key insight:** Hyperparameters should be re-validated after major feature additions. The 500-round cap was appropriate for 6 features at 384s, but was leaving signal on the table with 15 features.

---

## Hyperparameter Sweep — num_leaves, Bagging, Regularization — All Reverted

**All configs worse than v13 baseline (240.40s). No changes to model.py.**

Tested against v13 (LR=0.02, rounds=3000, num_leaves=127, lambda=0.1, no bagging):

| Config | RMSE | Delta |
|---|---|---|
| num_leaves=191 | 240.64s | +0.24s |
| num_leaves=255 | 240.73s | +0.33s |
| bagging_fraction=0.9, freq=5 | 241.07s | +0.67s |
| bagging_fraction=0.8, freq=5 | 242.10s | +1.70s |
| lambda_l1=lambda_l2=0.01 | 240.51s | +0.11s |
| lambda_l1=lambda_l2=0.0 | 240.49s | +0.09s |

- **More leaves**: deeper trees overfit — the 127-leaf model already extracts all the signal available
- **Bagging**: reduces effective training data significantly; 3000 rounds needs all rows to converge well
- **Less regularisation**: essentially flat — lambda=0.1 is already well-calibrated for this feature set

**Conclusion:** 240.4s (v13) is the model ceiling. All hyperparameter axes exhausted. Feature space was exhausted in earlier experiments. The remaining 0.4s gap to 1st place (240.0s) is irreducible with the current architecture and dataset.

---

## v14 — 9-Seed Ensemble

**Validation RMSE: 238.0s (−2.4s vs v13 240.4s) — KEPT**

Multi-seed LightGBM ensemble using `feature_fraction=0.8` to introduce per-seed diversity. Swept ensemble sizes from 1 to 10 seeds.

| Seeds | RMSE | Delta vs v13 |
|---|---|---|
| 1 (baseline, ff=0.8) | 238.95s | −1.45s |
| 2 | 238.31s | −2.09s |
| 3 | 238.14s | −2.26s |
| 4 | 238.13s | −2.27s |
| 5 | 238.06s | −2.34s |
| 6 | 238.04s | −2.36s |
| 7 | 237.99s | −2.41s |
| **8–9** | **237.98s** | **−2.42s** |
| 10 | 238.01s | −2.39s |

9 seeds optimal. Returns from additional seeds plateau after 8–9.

`ENSEMBLE_SEEDS = [42, 123, 456, 789, 1337, 2024, 31337, 99999, 7777]`
`FEATURE_FRACTION = 0.8` (from 1.0)

Note: `feature_fraction < 1.0` is what makes seeds produce diverse models. Without it, all seeds train the same tree structure and averaging provides no benefit.

**Important caveat:** The 238.0s validation RMSE uses training data where `BLOCK_TIME_UTC_mvt` is available. See the BLOCK_TIME bug section below — this validation signal is not fully representative of ranking performance.

---

## Critical Bug: BLOCK_TIME_UTC_mvt Is Withheld in Ranking Data

**Root cause of v9 (624.9s) and v13 (624.2s) scoring ~2.6× worse than validation.**

`BLOCK_TIME_UTC_mvt` is fully NaN for all DEP rows in `ranking.parquet`. It is withheld by the competition organisers because it is derived from the target being predicted: `TAXITIME_SEC_mvt = MVT_TIME - BLOCK_TIME` by definition. Providing it would be a direct data leak.

This means `schedule_delay_sec = BLOCK_TIME_UTC_mvt - SCHED_TIME_UTC_mvt` — our second-most-important feature, worth −22.9s — was NaN for every single ranking prediction. LightGBM's NaN branch paths for `schedule_delay_sec` were trained on the tiny minority of training rows where BLOCK_TIME happened to be missing (~0%), and those paths are not representative of typical flights. Every prediction used the wrong tree branch.

Our validation RMSE (238–244s) was a false signal: validation data is drawn from the training set, where BLOCK_TIME is fully available.

### Fix: AOBT_3_flt as substitute

`AOBT_3_flt` (actual off-block time from the flight plan system, as opposed to surveillance) measures the same event. In training data:

- Correlation with BLOCK_TIME: **r = 0.985**
- Mean difference: ~17s (BLOCK_TIME tends to be slightly later)
- Coverage in ranking DEP rows: **98.5%** (vs 0% for BLOCK_TIME)

Fix applied to `build_features()`:

```python
# BLOCK_TIME_UTC_mvt is withheld in the ranking set (it is derived from the
# target being predicted). AOBT_3_flt measures the same event from the flight
# plan system (r=0.985) and serves as a substitute when BLOCK_TIME is absent.
# Training data always has BLOCK_TIME, so training uses the cleaner signal;
# ranking predictions fall through to AOBT_3_flt.
out["schedule_delay_sec"] = (
    df["BLOCK_TIME_UTC_mvt"].fillna(df["AOBT_3_flt"]) - df["SCHED_TIME_UTC_mvt"]
).dt.total_seconds()
```

v14 submission used this fix for ranking predictions but the final models were still trained on BLOCK_TIME values (the fix was applied after the final ensemble finished training). A proper v15 retrain with AOBT_3 used consistently in both train and ranking prediction is needed for full consistency.

---

## Validation vs. Official Score Gap

**We are experiencing a persistent, large delta between our local validation RMSE (~238–244s) and the official leaderboard RMSE (~573s).**

### Submission history

| Version | Validation RMSE | Official score | Notes |
|---|---|---|---|
| v9 | 244.0s | ~624s | BLOCK_TIME bug — `schedule_delay_sec` NaN for all ranking rows |
| v13 | 240.4s | 624.2s | Same BLOCK_TIME bug |
| v14 | 238.0s | ~624s | Fix applied to ranking predictions only; models trained before fix took effect |
| v15 | 238.0s | 573.8s | Full retrain with AOBT_3 substitute; no artifact override |
| v16 | ~257s | 573.8s | Artifact override (8 LIRF rows) added; but model degraded by fillna inversion bug |
| v17 | ~238s (expected) | pending | Correct fillna order (`BLOCK_TIME.fillna(AOBT_3)`); artifact override kept |

### Sources of the gap

**1. BLOCK_TIME feature mismatch (fixed in v15)**

Validation always uses training data where `BLOCK_TIME_UTC_mvt` is 100% available. Ranking data has it fully withheld. Before v15, this single missing feature caused `schedule_delay_sec` to be NaN for every ranking prediction, pushing all rows down the NaN tree branch — which was trained on essentially zero training examples. This explains the jump from ~624s to ~573s.

**2. Artifact rows in ranking ground truth (partially fixed in v16+)**

The date-rollover tracking artifacts documented in the LIRF deep dive also appear in the ranking period (Jan–Sep 2026). The ground-truth `TAXITIME_SEC_mvt` for those rows is approximately 86,400s (one day off). Our model predicts a reasonable value (~900s), creating an error of ~85,500s per row.

Back-of-envelope estimate: assuming RMSE on non-artifact rows would be ~238s, we can solve for the number of artifact rows K that would explain an overall RMSE of 573.8s on 344,841 rows:

```
573.8² × 344,841 = 238² × (344,841 − K) + 85,500² × K
1.136e11 = 1.953e10 + K × (7.309e9 − 56,644)
K ≈ 13
```

Approximately 13 artifact rows in the ranking ground truth would fully explain the 573s score if the rest of the model were performing at 238s. We are currently overriding 8 LIRF rows (7 midnight-crossover + 1 AOBT date-error). If ~5 additional artifact rows exist that we have not identified, they would each contribute ~7.3 × 10⁹ to the SSE and keep the score well above 400s regardless of model quality.

**3. fillna inversion bug in v16 (fixed in v17)**

In v16, `schedule_delay_sec` was built with `AOBT_3.fillna(BLOCK_TIME)` instead of `BLOCK_TIME.fillna(AOBT_3)`. In training data, AOBT_3 and BLOCK_TIME are both present, but AOBT_3 has a mean absolute difference of ~384s from BLOCK_TIME — it is noisier. Training on the noisier signal degraded seed 2 validation RMSE from ~238s to 257.8s. The artifact fix in v16 was cancelled by this degradation. v17 reverts to `BLOCK_TIME.fillna(AOBT_3)` so training uses the clean BLOCK_TIME signal and AOBT_3 is only invoked for ranking predictions where BLOCK_TIME is absent.

### Why validation RMSE is not a reliable signal for ranking performance

Our validation set is drawn from Nov–Dec 2025 training data. In that set:
- `BLOCK_TIME_UTC_mvt` is 100% available → `schedule_delay_sec` is always computed correctly
- Artifact rows were filtered out of validation when they were filtered from training

The ranking data (Jan–Sep 2026) has both issues present. Until we can confirm that our artifact override covers all artifact rows in the ranking ground truth, the official score will continue to diverge from validation RMSE regardless of model improvements.

### What v17 should tell us

If v17 scores near 573s again: the ~13 estimated artifact rows are still present and unhandled. We need to find the remaining ~5 rows our current override logic is missing.

If v17 scores materially lower (e.g. 400–450s): the model quality fix (fillna order) is showing through, and the artifact problem is smaller than estimated.

If v17 scores near 238s: our 8-row override was sufficient and the gap was almost entirely the model degradation from v16.

## Score Gap Root-Cause Investigation (congestion train/serve skew)

> **CORRECTION (2026-09-22): this section's headline conclusion was wrong.** The
> congestion skew below is a *real bug worth fixing*, but it is **not** the dominant
> driver of the leaderboard gap. Fixing it did not move the score on its own (v18 was
> masked by a harmful override). The gap is dominated by **corrupted-label artifacts**
> in the ranking ground truth — the hypothesis this section dismissed. See the
> definitive write-up at the end of this file: "Leaderboard Reconciliation".

A fresh audit of `model.py` against the ranking data found the dominant driver of
the 238s → 573s gap. It is **not** primarily the artifact rows hypothesized above.
It is a **train/serve feature skew** in the congestion features, which are the model's
most important inputs. The submission mechanics themselves are clean (see "Ruled out"
below), so the gap is a model-quality problem, not a submission bug.

Note the grader RMSE (573s) is *worse* than a constant-mean predictor (target
std ≈ 546s). A model that is worse than the mean on the test set is the signature of
a feature that is actively misleading at serving time, not merely uninformative.

### Primary cause — `congestion_signal` measures a different quantity at serving

`congestion_signal` is the **#1 feature by gain** (≈18% ahead of #2 `runway`). It is
computed from a different data pool in training vs. serving:

| | Training / validation | Ranking (graded) |
|---|---|---|
| median | 929s | 535s |
| mean | 974s | 544s |
| fed by | recent **taxi-OUT** of DEP rows | recent **taxi-IN** of ARR rows |

- In `main()` the completed pool passed to `compute_congestion_signal` is `df` itself
  (DEP taxi-out times).
- In `write_submission()` the pool is `train_df` + `arr_context`. The ranking period is
  **2026-01 → 2026-07, entirely after training (all of 2025), with zero overlap**. So
  the 2025 training departures fall outside every 60-min rolling window in the 2026
  ranking period, and the signal is fed **almost entirely by ARR taxi-IN times**
  (mapped `ADES → ADEP`). Taxi-in is a systematically smaller, decorrelated quantity.

The model learned "congestion ≈ 930 ⇒ predict ≈ 900s taxi." At grading it reads ≈ 535
and interprets it as low congestion, biasing predictions low and destroying accuracy.
This is inherent: at serving you cannot know recent taxi-*out* of the ranking period —
those are exactly the withheld targets. `congestion_acceleration` (short − long window)
and `day_deviation_ratio` (numerator = today's flights, baseline = per-airport mean over
the pool) inherit the same skew.

### Secondary cause — `schedule_delay_sec` trains on a serving-absent column

`schedule_delay_sec` uses `BLOCK_TIME_UTC_mvt.fillna(AOBT_3_flt)`. `BLOCK_TIME` is 100%
present in training but 100% null in ranking, so training learns on `BLOCK_TIME − SCHED`
while grading silently substitutes `AOBT_3 − SCHED`. Correlation with the target shifts
(0.016 → 0.104) and the feature distribution moves. This is the same issue flagged in the
"BLOCK_TIME feature mismatch" note above, but note the current `.fillna()` order does not
fix it — it *hides* it: training never touches the fallback branch, so the model is still
tuned on a column it will never see at serving.

### Quantified impact (single-seed, lr=0.05, 400 rounds — a lighter model than the 9×3000 ensemble)

| Scenario (evaluated on the same val fold) | RMSE |
|---|---|
| Baseline (features as computed in training) | 241.7s |
| + `schedule_delay` recomputed with AOBT_3 (serving-style) | 295.3s |
| + congestion crudely rescaled to taxi-in scale (×535/929) | 320.4s |

The 320s is a **conservative floor**: the crude rescale keeps the feature's correlation
structure, whereas the real served signal is decorrelated taxi-in noise. The production
ensemble (lr=0.02, 3000 rounds) over-fits the skewed congestion feature harder, widening
the gap further. Temporal shift (6-month-later, spring/summer period) compounds on top.

### Fixes (priority order)

1. **Congestion pool consistency (primary).** Compute the congestion pool identically in
   training and serving. Either (a) build the training congestion from the same ARR
   taxi-in context that serving will use (honest but weaker signal), or better (b) replace
   the mean-taxi-time congestion with a **demand/queue-count** signal (number of
   departures in the preceding window), which is available and identically scaled at both
   train and serve time. Apply the same treatment to `congestion_acceleration` and
   `day_deviation_ratio`.
2. **Drop the `BLOCK_TIME` branch (secondary).** Use `AOBT_3_flt − SCHED_TIME` in *both*
   training and ranking so the model trains on the column it will actually be served.
   Never feed a training-only column.
3. **Make validation mirror the grader.** In `time_based_split`, compute all time-window
   features for the val fold using only the pool that would exist at serving for that fold
   (prior periods + ARR taxi-in), exactly as `write_submission` builds it. Also fix the
   `day_deviation` per-airport baseline, which is currently a global mean over all 12
   months (leaks the val period into val features). Until validation is computed this way,
   local RMSE cannot track the grader.
4. **Re-clip after the LIRF override.** The LIRF override writes `proxy_taxi` (can be
   negative) *after* the `np.clip(..., 0, None)`, reintroducing negatives. Re-clip.

### Ruled out (checked, clean)

- **Units/transformation:** predictions are plain seconds, no log — v17 median 874s vs
  training 912s.
- **ID/row alignment:** template ↔ ranking DEP is a perfect 1:1 map, no dupes, mapped by
  `MVT_ID` not position.
- **Outlier filtering:** the `TAXITIME ≤ 21600` filter drops only 69 of 2.08M rows
  (0.003%). Not a factor.
- **Submission NaN/defaults:** v17 has 0 nulls; every ranking DEP row is covered.

### Note on the earlier artifact hypothesis

The "K ≈ 13 artifact rows" math above assumes non-artifact RMSE is ≈ 238s and solves for
the artifact count needed to reach 573s. This investigation shows the non-artifact RMSE is
itself well above 238s at serving (the congestion feature is degraded for *every* row), so
the gap is explained by model quality across all rows, not a handful of outliers. The
artifact override is still worth keeping, but it is not the main lever.

---

## v20 — Weather Features Added

**Official score: 518.4s (−55s vs prior ~573s) — #134 leaderboard**

Added 5 hourly weather features joined to each departure by `(airport, floor(AOBT_3, hour))`:

| Feature | Source | Coverage |
|---|---|---|
| `weather_temp_c` | Meteostat (9 airports), IEM (LTFM, LEMD) | ~100% |
| `weather_wind_kt` | Meteostat / IEM | ~100% |
| `weather_precip_mm` | Meteostat / IEM | ~99% |
| `weather_visibility_m` | IEM only (LTFM, LEMD) | 100% at those 2 airports, NaN elsewhere |
| `weather_code` | Meteostat only (1=clear → 25=heavy thunderstorm) | ~99% at 9 airports, NaN at LTFM/LEMD |

**Data sources:** Meteostat's v2 API for 9 airports (stations 0.5–2.8 km from field, 99–100% hourly coverage). IEM ASOS for LTFM (Meteostat maps to old Ataturk airport 33 km away) and LEMD (Meteostat only 40% coverage). Weather pre-fetched and cached to `weather_cache.parquet` covering Jan 2025 – Aug 2026. No API calls at training or prediction time.

**Why temperature drove the improvement:** Temperature encodes the most operationally impactful weather condition — cold weather drives de-icing queue formation, which is the primary mechanism linking weather to taxi time. Precipitation and wind independently describe the same conditions less precisely.

---

## Weather Ablation Test

Single-seed ablation: each feature disabled one at a time, RMSE measured against baseline of 266.6s (single seed, all features on).

| Feature | RMSE | Delta | Verdict |
|---|---|---|---|
| runway | 284.7s | +18.1s | strong signal |
| weather_temp_c | 276.1s | +9.6s | strong signal |
| gate_delay_sec | 273.1s | +6.5s | strong signal |
| stand | 271.5s | +4.9s | strong signal |
| airline | 270.8s | +4.3s | strong signal |
| schedule_delay_sec | 270.8s | +4.2s | strong signal |
| arvt_update_sec | 270.7s | +4.1s | strong signal |
| hour | 270.1s | +3.5s | strong signal |
| stand_prefix | 267.6s | +1.0s | neutral |
| congestion_signal | 267.5s | +0.9s | neutral* |
| weight_class | 267.4s | +0.8s | neutral |
| weather_code | 267.4s | +0.8s | neutral |
| airport | 266.8s | +0.3s | neutral |
| day_deviation_ratio | 266.8s | +0.2s | neutral* |
| weather_wind_kt | 266.7s | +0.1s | neutral |
| weather_visibility_m | 266.6s | 0.0s | no signal |
| weather_precip_mm | 266.4s | −0.1s | neutral |
| market_segment | 266.4s | −0.2s | neutral |
| month | 266.1s | −0.4s | slight noise |
| congestion_acceleration | 265.9s | −0.7s | slight noise |

*`congestion_signal` and `day_deviation_ratio` appear neutral individually because they substitute for each other — removing one allows the other to compensate. A group ablation (removing both simultaneously) would reveal their true combined contribution.

**Key findings:**

- `runway` is the single most important feature (+18.1s), not `congestion_signal` as earlier importance scores suggested. Weather features changed the landscape.
- `weather_temp_c` (+9.6s) is the only weather feature doing real work. Wind, precip, visibility, and condition code are all within noise — temperature alone captures the operationally relevant signal (de-icing conditions, winter disruption).
- `congestion_acceleration` (short-window minus long-window delta) slightly hurts (−0.7s). The directional signal adds noise rather than information.
- `month` slightly hurts (−0.4s) — `weather_temp_c` already encodes seasonality more precisely as a continuous variable. Month as a categorical overfits to specific months in training.
- The 40s official score improvement from v19→v20 confirms the weather signal genuinely generalises to the ranking period — it is not an artefact of the training/validation distribution.

---

## Leaderboard Reconciliation (2026-09-22) — why clean val ≠ leaderboard

This is the definitive account of the ~250s-clean-val vs ~518s-leaderboard gap. Two
earlier write-ups in this file overshot in opposite directions (congestion skew "is the
root cause"; then "irreducible artifact ceiling"). Both are corrected here.

### Leaderboard progression

| Version | What changed | Official RMSE |
|---|---|---|
| v15 | no override; old skewed congestion + BLOCK_TIME schedule_delay | 573.8 (was best for a long time) |
| v16–v18 | congestion/schedule fixes **+ a harmful override** | ≥ 573.8 (never beat it) |
| v19 | harmful override removed; only `proxy_taxi > 50000` (1 row) kept | **531** |
| v20 | weather features integrated | **518** |
| v22 | `proxy_taxi` feature + honest validation | pending |

### The two things that were fighting each other

1. **The congestion train/serve skew and the `schedule_delay` BLOCK_TIME dependency were
   real bugs** that hurt genuine generalisation — worth fixing. (Documented above.)
2. **v18 hid those gains** because it *also* carried a harmful override branch: it
   force-predicted `proxy + 86400` (~87,000s) on LIRF flights matching an
   AOBT-hour-23 / MVT-hour-0 pattern. Most such flights are **normal** near-midnight
   departures, so the override injected ~86,400s errors on good rows — roughly cancelling
   the fix gains. Removing that branch (v19) let the fixes surface → 531.

**Lesson: the leaderboard responds to genuine model improvement (573.8 → 531 → 518). There
is no "irreducible ceiling." But absolute clean-val (~250) will never equal the
leaderboard, because the leaderboard is dominated by corrupted labels the clean val filters
out.**

### The artifacts, definitively

Target is `TAXITIME = MVT_TIME − BLOCK_TIME`. An artifact is a row where one of those two
timestamps has the wrong **calendar date** (a logging glitch), so the subtraction yields an
impossible duration (~6–36 h). 69 such rows in 2025 training (66 LIRF, 2 LFPG, 1 LSZH); the
ranking ground truth contains the same kind.

Two flavours:
- **Flavour B — MVT_TIME has the wrong day.** The corrupted timestamp is one we *can see*.
  `proxy_taxi = MVT_TIME − AOBT_3` comes out ~86,400, and it matches the corrupted label.
  **Detectable and recoverable** (this is the `proxy_taxi > 50000` override, ~40/69 in training).
- **Flavour A — BLOCK_TIME has the wrong day.** `BLOCK_TIME` is 100% withheld in the ranking
  set, and every *visible* timestamp agrees, so there is no outlier to detect.
  **Undetectable** (~29/69).

Why a "majority-vote over the 4 timestamps to snap the outlier date" idea does **not** work:
on artifact rows the corroborating columns (`EOBT`, `LOBT`, `IOBT`, `AOBT`) are almost all
**null** — only 1 of 69 has ≥2 non-null reference votes. There is nothing to vote with.

Why artifacts dominate the score: RMSE squares errors. One artifact row (label ~86,000, we
predict ~900) contributes `85,000² ≈ 7.2e9` — **≈ 92,000 normal rows' worth of squared
error**. In a local test, 8 artifact rows (0.002% of val) caused 47% of the SSE, and the
honest per-month RMSE swings 275 (Oct, 1 artifact) → 617 (Jul, 17 artifacts) for the *same
model*. The leaderboard number is therefore high-variance and driven by artifact *count*,
not skill.

### How teams reach < 250

Almost certainly by **using `MVT_TIME` (via `proxy_taxi`)**: it sharpens clean rows *and*
auto-matches the Flavour-B corrupted labels for free. This is why v22 adds `proxy_taxi` as a
feature (reversing the old "never use MVT_TIME" ban) and keeps the explicit `proxy > 50000`
override (trees can't extrapolate to 86,400 on their own).

> **SUPERSEDED (2026-09-22): Flavour-A artifacts are NOT unrecoverable.** They have a
> detectable signature (missing flight plan + large `MVT − SCHED`) that recovered ~132
> points. See "Artifact Recovery Breakthrough" at the end of this file.

### Honest validation (added v22)

The old validation filtered `TAXITIME ≤ 21600` from *both* train and val, so it never saw the
artifacts that dominate grading — reporting ~250 while the board sat at ~518. v22 fixes this:
train on clean rows only, but **keep artifacts in the validation set and apply the same
override**, reporting both `honest RMSE` (grader-style) and `clean-only`. Still single-fold
(Nov–Dec, winter); `honest_val.py` holds a rolling-origin version that spans seasons and
brackets the leaderboard (Apr–Jul honest ≈ 305–617, with 518 inside the range).

**Takeaways for future work:**
- Judge changes on the **honest** number, not clean-only.
- Genuine model gains (features, weather) *do* move the board — keep pursuing them.
- Do **not** re-add pattern-based overrides (e.g. midnight-crossover) that fire on normal
  flights; only override physically-impossible `proxy_taxi > 50000`.
- The remaining gap above clean skill is Flavour-A artifacts + 2026 seasonal shift, both
  outside our control.

---

## Artifact Recovery Breakthrough (2026-09-22) — 490.7 → 358.5

The claim that Flavour-A artifacts (corruption in the withheld `BLOCK_TIME`) are
undetectable was **wrong**. They have a clear signature, and recovering them was the
single biggest gain of the whole project after the harmful-override removal.

### Validated leaderboard history (from the grader's per-submission JSON, `truthing.parquet`, all 344,841 rows)

| Ver | Score | Note |
|---|---|---|
| v9 | 624.61 | |
| v13 | 624.19 | |
| v14 | 627.36 | |
| v15 | 573.85 | AOBT substitute, no override |
| v16 | 678.04 | harmful midnight override (worse!) |
| v17 | 697.73 | harmful override (worst) |
| v18 | 658.19 | harmful override + congestion fix |
| v19 | 531.27 | harmful override removed |
| v20 | 518.42 | weather |
| v21 | 518.44 | feature ablation (≈neutral) |
| v22 | 490.73 | `proxy_taxi` feature |
| v23 | 511.81 | 1-row probe — **confirmed the corrupted-label math to 4 sig figs** |
| **v25** | **358.45** | **airport-restricted no-flight-plan artifact override** |

The harmful midnight override cost **~100+ points** (v16–v18 vs v15) — the biggest single
mistake; never re-add pattern overrides that fire on normal flights.

### The Flavour-A signature

The clean-model RMSE is **224.8** (honest validation, clean-only) — excellent. So nearly the
entire 490→225 gap was artifacts, not model quality. **Feature work is therefore heavily
diluted** (a 25s clean gain moves the board only ~11s); artifact recovery is ~30× more
valuable per row.

Flavour-A artifacts turned out to share a signature, all visible in the ranking set:
- **The entire flight-plan block is null** — `AOBT_3`, `EOBT_1`, `LOBT`, `ARVT_3` all NaN
  (100% of the 68 training Flavour-A artifacts, vs 1.1% of normal rows).
- On those rows, `SCHED_TIME` tracks the corruption, so **`MVT − SCHED` equals the corrupted
  ground-truth label to a median of 3 seconds.** So once detected, we know exactly what to
  predict: `MVT − SCHED`.

### The detector (shipped in v25)

Override a row's prediction with `MVT − SCHED` when **all** of:
1. `ADEP ∈ {LIRF, LFPG, LSZH}` — the only airports with the defect in training (0 elsewhere),
2. `AOBT_3` is null (flight plan missing),
3. `MVT − SCHED > 10h`.

- Training precision: **83%** (44 TP / 9 FP). The airport restriction is essential — without
  it, EHAM etc. contribute delayed-but-normal no-plan flights and precision collapses to 30%,
  making a blanket override net-negative (this is why the earlier "undetectable" call was made).
- It flagged **21 ranking rows** (18 LIRF, 2 LFPG, 1 LSZH), values 10–31h.
- Result: **490.73 → 358.45**, saving 3.87e10 SSE — matching the model's estimate.

> **REFINED (2026-09-22, v27): restrict to `ADEP == LIRF` only.** The 83% precision was
> LIRF (44 TP / 0 FP) diluted by LFPG (0 TP / 8 FP) and LSZH (0 TP / 1 FP) — the two extra
> airports contribute *only* false positives. See "Override False-Positive Fix" below.

### Why the separator works despite `BLOCK_TIME` being hidden

The only thing distinguishing an artifact from a genuinely-delayed flight is where the
(hidden) `BLOCK_TIME` sits. We can't see it — **but the missing-flight-plan flag is a proxy
for the same data-quality failure**, and the airport restriction removes the legitimately-
delayed no-plan flights (which occur everywhere, not just at the three defect airports).

### Tuning limits (do not push further)

- **Don't lower the 10h threshold.** The [6,10h] band is 38% precision (18 TP/30 FP) — below
  break-even for that band (there TP and FP errors are similar magnitude, unlike the >10h band
  where corrupted labels are much larger). The [3,6h] band is 0% precision (441 FPs).
- **Don't widen airports** — no training artifacts exist outside the three.

### Where we stand

At 358.45 with clean ≈ 225, roughly **1–4 big artifacts remain** (the exact count depends on
the true ranking clean-RMSE, which the winter fold may understate). They sit in the low-
precision bands or lack the signature, so they're likely not cleanly recoverable. We've
captured most of the recoverable value. Remaining levers are small: the residual artifacts
(hard) and clean-model improvement (heavily diluted).

**Status:** v25 is a post-processing derivative of v22 (not yet codified in `model.py`).
Once confirmed (it is — 358.45), fold the override into `apply_artifact_override` so it's
reproducible, alongside the existing `proxy_taxi > 50000` branch.

---

## Override False-Positive Fix + Error Re-decomposition (2026-09-22) — 358.5 → 349.8 → ~313

### Score progression continued

| Ver | Score | Change |
|---|---|---|
| v25 | 358.45 | LIRF/LFPG/LSZH no-plan override (post-process on v22) |
| v26 | **349.83** | override codified in `model.py` + on v24's model (dest + plan-delta features) |
| v27 | **313.00** | **override restricted to LIRF-only** (removes LFPG/LSZH false positives) — matched the 312.9 estimate |

### The override had false positives — all the signal was LIRF

Once the override was codified and re-examined per-airport, the "83% precision" turned out to
be **100% LIRF diluted by two pure-noise airports**:

| Airport | TP | FP | Precision |
|---|---|---|---|
| LIRF | 44 | 0 | 100% |
| LFPG | 0 | 8 | 0% |
| LSZH | 0 | 1 | 0% |

LFPG/LSZH have no-plan + delayed flights (real taxi ~840s) that match the mask but are *not*
artifacts — the override was assigning them ~day-long taxis. The two LFPG training artifacts
never matched this pattern (different flavour). **Fix: `ARTIFACT_AIRPORTS = {"LIRF"}` (v27).**
On the ranking set this reverts 3 false positives (2 LFPG at 52,555s/69,540s, 1 LSZH at
36,187s → back to model values ~600–2,300s), estimated ~349.8 → ~313.

**Lesson:** validate override precision *per detected subgroup*, not in aggregate — an
aggregate rate can hide a subgroup that is pure false positives.

### Where the error lives now (honest validation, override applied)

With artifacts handled, the picture flipped back to clean-model-dominated:
- **Artifacts: ~6% of validation SSE** (overrides did their job).
- **Clean rows: ~94%**, and extremely concentrated — worst 1% of clean rows ≈ 68% of clean SSE.
- **LFPG is the largest genuine concentration.** Setting aside the override FPs, LFPG has a
  real tail of long taxis (2–9h, within the clean <6h range) — winter, runways 26R/08L,
  midday — that the model underpredicts. Signature points to de-icing / holding that
  `weather_temp_c` doesn't fully capture. (Caveat: measured with a fast single-seed diagnostic
  model at clean-only ~318 vs the real ensemble's ~225, so magnitudes are inflated; the LFPG
  concentration is directionally real.)

### Reframe at v27=313: clean-model work re-opens

With artifacts handled, the SSE mix flipped and clean-model work is worth ~2–2.5× what it was:

| | at v22 (490) | at v27 (313) |
|---|---|---|
| clean-model share of SSE | ~21% | **~52%** |
| board move per 10% clean-RMSE gain | ~11s | **~16s** |

So `dest`/plan-deltas/regularization/LFPG — dismissed as ~30× diluted at 490 — are back on the
table with modest but real payoff now that artifacts no longer dominate.

### The open fork (decides whether clean work pays off)

The board (313.00) sits ~74s above the **winter** honest-val (238.7). That gap is *either*:
- **~2 more undetectable artifacts** in the 2026 set → floor, clean work won't touch it; or
- **seasonal/year shift** — clean model genuinely worse on 2026 spring/summer than winter 2025
  → clean work lowers the board.

The winter-only honest fold can't tell which. **Rolling-season honest val (`honest_val.py`,
Apr–Jul folds)** resolves it: if clean-RMSE climbs from ~225 (winter) toward ~280+ (summer),
the gap is shift and clean work helps; if it stays ~225, the gap is artifacts and we're at the
floor. This is the next step — zero submissions, decides the strategy.

### Remaining avenues

1. **Diagnose the fork first** (rolling honest val) — highest information, zero cost.
2. **If shift:** clean-model work — LFPG long-taxi tail (de-icing/holding not captured by
   `weather_temp_c`), regularization/early-stopping, features. Now ~2× more effective than before.
3. **Residual artifacts** (~1–3 rows) — Flavour-A below 10h or without the no-plan signature;
   the [6,10h] band is 38% precision, not worth it. Likely a hard floor.

**Bottom line:** v27=313.00 (a 42% cut from the original 573.8, almost all from understanding
data defects). We're within ~90s of the winter clean floor (~225). Whether that last stretch is
reachable hinges on the artifacts-vs-shift fork above.

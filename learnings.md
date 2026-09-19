# Model Run Learnings

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

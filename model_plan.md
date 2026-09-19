# Taxi Time Prediction — Model Plan

## Target Variable

`TAXITIME_SEC_mvt` — taxi time in seconds. Predict in seconds, evaluate in seconds.

---

## Scope

Train and predict on **departures only** (`PHASE_mvt == 'DEP'`). The submission dataset contains only taxi-out times, so arrival rows add noise and complexity without benefit. Filter to departures before any feature engineering or model training.

---

## Features

### Include

| Feature | Source column(s) | Notes |
|---|---|---|
| Airport | `ADEP_mvt` | Large effect (ε²=0.35–0.40). Departures only, so always `ADEP_mvt`. |
| Runway | `RUNWAY_mvt` | Large effect (ε²=0.39). Median ranges from ~8 to ~23 min depending on runway. |
| Stand/gate | `STAND_mvt` | Large to medium effect per airport (ε²=0.10–0.20). Some stands are 2× slower than others. |
| Airline | `AIRCRAFT_OPERATOR_flt` | Medium effect (ε²=0.11). Consistent across airports — not just an airport mix artifact. Fastest airlines ~4 min below baseline, slowest ~6 min above. 11 airlines in ranking have no training history; fall back to airport baseline for those. |
| Aircraft weight class | `WK_TBL_CAT_flt` | Small effect (ε²=0.028). J=16 min, H=14 min, M=11 min, L=8 min median. Better proxy than raw aircraft type. |
| Market segment | `MARKET_SEGMENT_flt` | Small effect (ε²=0.026). Worth including given low complexity. |
| Month | `MVT_TIME_UTC_mvt` → month | Negligible ε² but real systematic bias of ~1 min peak-to-trough. Include as an ordinal or cyclical feature. |
| Rolling congestion signal | Derived from `MVT_TIME_UTC_mvt` + `TAXITIME_SEC_mvt` | Rolling mean deviation of recent completed flights at same airport. Window size defined as `CONGESTION_WINDOW_MINUTES = 60` — easy to tune. Spearman r≈0.23 overall, much stronger on disruption days. Include and validate via CV. |
| Gate delay | `AOBT_3_flt` - `EOBT_1_flt` | Confirmed. Late gate departure adds ~2 min taxi time (Pearson r=0.25, ~6% variance explained). Use continuous gap value, not binary flag. |

### Exclude

| Feature | Reason |
|---|---|
| Day of week | ε²=0.0001 — negligible, confirmed |
| Hour of day / time block | ε²=0.009 — negligible |
| Raw aircraft type (`AIRCRAFT_TYPE_mvt`) | 182 levels, noisy. Use `WK_TBL_CAT_flt` instead. |
| `FLIGHT_RULE_mvt` | 99.9% IFR — no variance |
| `PHASE_mvt` / arrival rows | Predicting departures only — phase is constant, arrivals are excluded from training |
| `ADES_mvt` | Arrival airport irrelevant for taxi-out time |

### Consider / Investigate Further

| Feature | Question |
|---|---|
| Stand prefix / terminal area | ε²=0.05 (medium). Could be useful if `STAND_mvt` is missing at prediction time. |
| Stand prefix / terminal area | ε²=0.05 (medium). Could be useful if `STAND_mvt` is missing at prediction time. |
| Flight type (`FLIGHT_TYPE_flt`) | ε²=0.002 — negligible, but near-zero complexity cost. Low priority. |

---

## Disruption Days

~1% of days show severe disruption (taxi times 2–5× normal). Three patterns observed:
- **Weather** (e.g., LTFM Feb 23): All-day cascade lasting 15+ hours. Rolling congestion feature will help.
- **Congestion** (e.g., LIRF Jul 13): Multi-hour waves, sometimes multiple incidents per day. Rolling feature helps.
- **Single incident** (e.g., LSZH Aug 12): One sharp hourly spike, no cascade. Rolling feature less useful here.

Model will systematically underpredict disruption days without the rolling feature. Include it and validate via CV.

---

## Encoding Strategy

| Feature type | Approach |
|---|---|
| Low-cardinality categoricals (phase, weight class, market segment, month) | One-hot or ordinal |
| High-cardinality categoricals (airport, runway, stand, airline) | Target encoding with shrinkage, or leave to a tree-based model to handle natively |
| Unknown airlines at prediction time (11 airlines, 1.1% of ranking) | Fall back to airport baseline (zero deviation) |
| Rolling congestion signal | Continuous, computed at prediction time using completed flights earlier that day |

---

## Validation Strategy

- **CV scheme**: time-based split (train on earlier months, validate on later months). Do not use random CV — taxi time has temporal autocorrelation.
- **Metric**: RMSE in seconds (captures disruption day errors, which matter most).
- **Feature ablation**: run CV with/without rolling congestion feature to confirm it reduces error before keeping it.

---

## Open Questions

- Is there interaction between airport × runway that should be modelled explicitly?
- Should airline be included as raw target-encoded feature, or grouped (e.g., by alliance/region) to reduce sparsity?

---

## Next Features to Build

Build and test one at a time. Add each to `build_features()` in `model.py`, run `model.py`, record RMSE delta in `learnings.md`. Revert if neutral or worse.

### 1. Departure queue length

**Hypothesis:** More departures scheduled in the same time window at the same airport = longer taxi (more planes waiting to line up).

**Implementation:**
- Source: `EOBT_1_flt` (scheduled pushback time) and `ADEP_mvt`
- For each departure, count other departures at the same airport with `EOBT_1_flt` within a ±30-minute window
- Vectorise with `searchsorted` on sorted scheduled times per airport — same pattern as `compute_congestion_signal`
- For ranking data: pool training DEP rows + ranking DEP rows as the "scheduled flights" source
- Output: integer count, feature name `departure_queue`
- Tune window size (±15, ±30, ±60 min) only if the feature shows clear signal

### 2. Scheduled departure hour

**Hypothesis:** `EOBT_1_flt` (what the airline planned) reflects structural schedule patterns more cleanly than `MVT_TIME_UTC_mvt` (the actual noisy pushback time).

**Implementation:**
- Source: `EOBT_1_flt`
- `df["EOBT_1_flt"].dt.hour` — numeric 0–23, same as the existing `hour` feature
- Replace the existing `hour` feature (which uses `MVT_TIME_UTC_mvt`) with this, or add as a second feature and let the model pick
- NaN when `EOBT_1_flt` is missing — LightGBM handles natively

### 3. Gate delay sign (is_late)

**Hypothesis:** The effect of gate delay may be asymmetric — late departure might force a faster/prioritised taxi (ATC sequence disruption), while early departure has different dynamics. A boolean flag lets the model learn separate intercepts.

**Implementation:**
- Source: existing `gate_delay_sec` (already in `build_features`)
- `out["is_late_departure"] = (gate_delay_sec > 0).astype(int)` — 1 if departed after scheduled time, 0 if on time or early
- Do not remove `gate_delay_sec`; this is additive

### 4. Stand-level rolling signal

**Hypothesis:** Some stands have physical bottlenecks (long taxi routes, shared apron). A stand-level congestion signal captures that on top of the airport-wide signal.

**Implementation:**
- Same vectorised approach as `compute_congestion_signal` but group by `(ADEP_mvt, STAND_mvt)` instead of just `ADEP_mvt`
- Extract the per-airport loop into a shared helper; call it twice — once for airport-level (existing), once for stand-level
- Fall back to NaN when no completed flights at that stand in the window (low-traffic stands) — LightGBM handles it
- Feature name: `stand_congestion_signal`
- Only worthwhile if stand has reasonable coverage; check coverage % before keeping

### 5. Day of week

**Hypothesis:** Monday morning push and Friday/Sunday leisure travel may show systematic queue patterns not captured by hour alone.

**Implementation:**
- Source: `MVT_TIME_UTC_mvt`
- `df["MVT_TIME_UTC_mvt"].dt.dayofweek` — numeric 0 (Mon) to 6 (Sun)
- Treat as categorical (7 levels, each day gets its own leaf splits)
- Previously excluded (ε²=0.0001 standalone), but may interact with congestion or airport features inside the tree

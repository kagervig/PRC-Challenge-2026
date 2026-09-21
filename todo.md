# Todo — RMSE Reduction Plan

Baseline: **244.0s** (current best, v9)
See `plan.md` for full context and implementation prompts.

---

## Improvement 1: Huber Loss

- [x] Step 1.1 — Add `alpha` param to `train()`
- [x] Step 1.2 — Run alpha sweep [None, 50, 100, 200, 400], print table
- [x] Step 1.3 — All alphas worse than MSE. Reverted. No change to main().

## Improvement 3: airport_runway Combined Categorical

- [x] Step 3.1 — Add `airport_runway` to `build_features()` and `CATEGORICAL_FEATURES`
- [x] Step 3.2 — Test variants A/B/C — all neutral or worse
- [x] Step 3.3 — Reverted. No change to model.py.

## Improvement 2: LFPG Bias Fix

- [x] Step 2.1 — Diagnostic complete. Root cause: runway distribution shift + hour pattern shift between train/val
- [x] Step 2.2 — Implemented airport_hour interaction
- [x] Step 2.3 — +0.4s worse. Reverted. LFPG bias not recoverable with available features.

---

## Results Log

| Step | Change | RMSE (s) | Delta | Decision |
|---|---|---|---|---|
| baseline | v9 current best | 244.0 | — | — |
| Huber best (alpha=400) | Huber loss | 246.1 | +2.1 | Reverted |
| airport_runway (best) | combined categorical | 244.0 | 0.0 | Reverted (neutral) |
| airport_hour | LFPG bias fix | 244.4 | +0.4 | Reverted |

**All three plan experiments failed to improve on 244.0s. Model is at ceiling for this feature set.**

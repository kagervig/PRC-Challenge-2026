# Todo — RMSE Reduction Plan

Baseline: **244.0s** (current best, v9)
See `plan.md` for full context and implementation prompts.

---

## Improvement 1: Huber Loss

- [ ] Step 1.1 — Add `alpha` param to `train()`
- [ ] Step 1.2 — Run alpha sweep [None, 50, 100, 200, 400], print table
- [ ] Step 1.3 — Wire best alpha into `main()`, update version, record in learnings.md

## Improvement 3: airport_runway Combined Categorical
_(done before LFPG — may resolve the bias diagnosis)_

- [ ] Step 3.1 — Add `airport_runway` to `build_features()` and `CATEGORICAL_FEATURES`
- [ ] Step 3.2 — Test variants A/B/C (keep/drop airport, runway)
- [ ] Step 3.3 — Wire best variant, update version, record in learnings.md

## Improvement 2: LFPG Bias Fix

- [ ] Step 2.1 — Run diagnostic: LFPG residuals by runway, stand prefix, hour, month
- [ ] Step 2.2 — Implement fix based on findings
- [ ] Step 2.3 — Run model, record delta, revert if worse

---

## Results Log

| Step | Change | RMSE (s) | Delta | Decision |
|---|---|---|---|---|
| baseline | v9 current best | 244.0 | — | — |

"""Summer-weighted rolling-season tuning harness.

The single winter time-split (model.main's honest val) understates congestion/disruption
features: the board (Jan-Jul) is dominated by summer surface congestion the winter fold
under-samples. This harness evaluates on rolling Apr-Jul folds (train on prior clean
months, eval on the target month with artifacts kept + the same override the submission
applies), row-weighted into one pooled RMSE — the metric that tracks the board.

It is the shared measurement tool for the feature-tuning phases. prepare() accepts
per-feature window overrides so Phase 1 can decouple the currently-shared 60-min window;
evaluate() runs a paired N-seed comparison so deltas are not seed noise.

Config is a deliberately fast proxy (lr=0.05, fixed rounds, no early stopping to avoid
eval leakage). Absolute RMSE differs from the real lr=0.02/3000-round model, but relative
deltas rank correctly; confirm any adopted change at full config before submitting.
"""
import gc

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import root_mean_squared_error

import model

SEEDS = [42, 123, 456]
EVAL_MONTHS = [4, 5, 6, 7]          # summer-weighted; the board's hard months
AIRPORTS = ["LFPG", "LIRF", "EGLL"]
NUM_ROUNDS = 600
PARAMS = dict(objective="regression", metric="rmse", num_leaves=127,
              learning_rate=0.05, min_data_in_leaf=50, lambda_l1=0.1, lambda_l2=0.1,
              feature_fraction=model.FEATURE_FRACTION, verbose=-1)


def load():
    """Load movements once and return (dep, pool, weather)."""
    mov = model.load_movements()
    dep = mov[mov["PHASE_mvt"] == "DEP"].copy()
    pool = model.build_congestion_pool(mov)
    weather = model.load_weather_cache()
    return dep, pool, weather


def prepare(dep, pool, weather, extra=None):
    """
    Build the full feature matrix, replicating model.main()'s current (v32) sequence.

    Mirrors the live model: EWMA congestion, recent_delay=15min, airport + runway queues.
    `extra` is a dict of {column: Series} merged in after build_features for testing new
    candidate features (e.g. Phase 4 runway-reconfiguration signals).
    """
    cong = model.compute_congestion_ewma(dep, pool, model.CONGESTION_EWMA_HALFLIFE_MIN)
    cong_short = model.compute_congestion_signal(dep, pool, model.CONGESTION_WINDOW_SHORT_MINUTES)
    dayd = model.compute_day_deviation_ratio(dep, pool)
    recent = model.compute_recent_delay(dep, model.RECENT_DELAY_WINDOW_MINUTES)
    arrdem = model.compute_arrival_demand(dep, pool, model.CONGESTION_WINDOW_MINUTES)
    queue = model.compute_departures_queue(dep)
    runway_queue = model.compute_runway_queue(dep)
    sched = model.compute_scheduled_push_density(dep)
    lead_wake = model.compute_lead_wake_category(dep)
    feats = model.build_features(dep, cong, dayd, cong_short - cong, weather,
                                 recent, arrdem, queue, sched, runway_queue, lead_wake)
    if extra:
        for col, series in extra.items():
            feats[col] = series
    return feats


def evaluate(feats, dep, label, seeds=SEEDS, verbose=True):
    """
    Paired N-seed rolling Apr-Jul evaluation. Returns (pooled_honest, pooled_clean).

    For each eval month: train on prior clean months (all seeds), average predictions,
    apply the artifact override, then pool (y, pred) across all folds for one row-weighted
    RMSE. Prints per-month and per-airport breakdowns.
    """
    target = dep["TAXITIME_SEC_mvt"].astype(float)
    is_clean = (target <= model.ARTIFACT_TAXI_MAX_SEC).values
    mo = dep["MVT_TIME_UTC_mvt"].dt.month.values
    # Guard against prepare() drifting behind model.py: only pass categoricals that were
    # actually built, and warn loudly so the missing feature gets noticed (not silently dropped).
    cats = [c for c in model.CATEGORICAL_FEATURES if c in feats.columns]
    missing = set(model.CATEGORICAL_FEATURES) - set(cats)
    if missing:
        print(f"  WARNING: categoricals in model but missing from prepare(): {missing}")

    all_y, all_adj, all_clean, all_ap = [], [], [], []
    per_month = []
    for M in EVAL_MONTHS:
        tr = (mo < M) & is_clean
        va = mo == M
        if tr.sum() < 1000 or va.sum() == 0:
            continue
        preds = []
        X_tr, y_tr, X_va = feats[tr], target[tr].values, feats[va]
        for s in seeds:
            p = dict(PARAMS, seed=s)
            # free_raw_data=True lets LightGBM release the training matrix after the
            # booster is built; predict() reads X_va, not the Dataset — so this is safe
            # and keeps peak memory bounded across the sweep's many fits.
            ds = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cats, free_raw_data=True)
            b = lgb.train(p, ds, num_boost_round=NUM_ROUNDS)
            preds.append(b.predict(X_va))
            del ds, b
            gc.collect()
        del X_tr, X_va
        mean_pred = pd.Series(np.mean(preds, axis=0), index=dep.index[va])
        adj = model.apply_artifact_override(mean_pred, dep[va]).values
        y = target[va].values
        cl = is_clean[va]
        m_honest = root_mean_squared_error(y, adj)
        m_clean = root_mean_squared_error(y[cl], adj[cl])
        per_month.append((M, va.sum(), m_honest, m_clean))
        all_y.append(y); all_adj.append(adj); all_clean.append(cl)
        all_ap.append(dep.loc[va, "ADEP_mvt"].values)

    y = np.concatenate(all_y); adj = np.concatenate(all_adj)
    cl = np.concatenate(all_clean); ap = np.concatenate(all_ap)
    pooled_honest = root_mean_squared_error(y, adj)
    pooled_clean = root_mean_squared_error(y[cl], adj[cl])

    if verbose:
        print(f"\n=== {label} ===")
        print(f"{'month':>6} {'n':>8} {'honest':>9} {'clean':>9}")
        for M, n, h, c in per_month:
            print(f"{M:>6} {n:>8} {h:>9.1f} {c:>9.1f}")
        print(f"{'POOL':>6} {len(y):>8} {pooled_honest:>9.2f} {pooled_clean:>9.2f}")
        print("  per-airport (pooled, honest / clean):")
        for a in AIRPORTS:
            sel = ap == a
            h = root_mean_squared_error(y[sel], adj[sel])
            c = root_mean_squared_error(y[sel & cl], adj[sel & cl])
            print(f"    {a}: {h:.1f} / {c:.1f}  (n={sel.sum()})")
    return pooled_honest, pooled_clean


if __name__ == "__main__":
    print("Loading data + computing baseline (current v30) feature set...")
    dep, pool, weather = load()
    feats = prepare(dep, pool, weather)
    print(f"  dep rows: {len(dep):,}  features: {list(feats.columns)}")
    evaluate(feats, dep, "BASELINE (current model, summer-weighted Apr-Jul)")

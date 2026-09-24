"""Diagnostic: is our model error concentrated on ATFM-disrupted airport-days?

This uses the daily airport-level pre-departure (ATFM) delay from Eurocontrol NM purely to
ATTRIBUTE model error — it never enters the model or a prediction, so there is no look-ahead
or leakage (the same-day-total look-ahead problem only exists when the value flows into a
prediction; here it flows into our understanding of the residuals).

It answers three things:
  1. Coverage — how many val flights join to an airport-day in the delay file.
  2. Attribution — bucket val flights by daily delay intensity (delay-min per departure) and
     report per-bucket RMSE, mean signed residual, and share of total squared error. Shows
     WHERE the error lives and whether we systematically UNDERpredict on disrupted days.
  3. The decisive test — does the daily delay explain residual variance the model does NOT
     already capture? If yes, an ATFM feature has untapped value and hunting the timestamped/
     per-flight version is worth it. If the residual is uncorrelated with daily delay, then
     even a perfect day-level regulation label would not help, and we should stop chasing it.

Residuals come from the same rolling Apr-Jul folds as the tuning harness, at the current
(v34) feature set. resid = pred - y  (positive = overprediction, negative = underprediction).
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import root_mean_squared_error

import model
import tune_harness as th

CSV = os.environ.get(
    "PREDEP_CSV", str(Path.home() / "Downloads" / "all_pre_departure_delays_2026.csv")
)
SEEDS = [42, 123]           # diagnostic — residual attribution does not need the full ensemble
N_BUCKETS = 10


def collect_val_residuals(dep, feats):
    """Rolling Apr-Jul folds; return a frame of per-flight val residuals with airport + local date."""
    target = dep["TAXITIME_SEC_mvt"].astype(float)
    is_clean = (target <= model.ARTIFACT_TAXI_MAX_SEC).values
    mo = dep["MVT_TIME_UTC_mvt"].dt.month.values
    cats = [c for c in model.CATEGORICAL_FEATURES if c in feats.columns]

    ref = dep["AOBT_3_flt"].fillna(dep["MVT_TIME_UTC_mvt"])
    local_date = pd.Series(pd.NaT, index=dep.index)
    for apt, tz in model.AIRPORT_TIMEZONES.items():
        m = (dep["ADEP_mvt"] == apt).values
        if m.any():
            local_date.loc[m] = ref[m].dt.tz_convert(tz).dt.date.values

    parts = []
    for M in th.EVAL_MONTHS:
        tr = (mo < M) & is_clean
        va = mo == M
        if tr.sum() < 1000 or va.sum() == 0:
            continue
        preds = []
        for s in SEEDS:
            p = dict(th.PARAMS, seed=s)
            ds = lgb.Dataset(feats[tr], label=target[tr].values,
                             categorical_feature=cats, free_raw_data=True)
            b = lgb.train(p, ds, num_boost_round=th.NUM_ROUNDS)
            preds.append(b.predict(feats[va]))
        mean_pred = pd.Series(np.mean(preds, axis=0), index=dep.index[va])
        adj = model.apply_artifact_override(mean_pred, dep[va]).values
        y = target[va].values
        parts.append(pd.DataFrame({
            "airport": dep.loc[va, "ADEP_mvt"].values,
            "date": local_date[va].values,
            "y": y,
            "pred": adj,
            "resid": adj - y,
        }))
        print(f"  fold M={M}: {va.sum():,} val rows", flush=True)
    return pd.concat(parts, ignore_index=True)


def load_predeparture(csv):
    df = pd.read_csv(csv, comment=None)
    df = df.rename(columns={"APT_ICAO": "airport", "FLT_DATE": "date",
                            "DLY_ALL_PRE_2": "dly_pre", "FLT_DEP_1": "deps"})
    df["date"] = pd.to_datetime(df["date"]).dt.date
    for c in ("dly_pre", "deps"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # delay-minutes per departure: intensity comparable across airports of different size
    df["dly_per_dep"] = df["dly_pre"] / df["deps"].replace(0, np.nan)
    return df[["airport", "date", "dly_pre", "deps", "dly_per_dep"]]


def main():
    if not Path(CSV).exists():
        sys.exit(f"pre-departure CSV not found: {CSV} (set PREDEP_CSV=/path)")

    print("Loading data + building v34 feature matrix...", flush=True)
    dep, pool, weather = th.load()
    feats = th.prepare(dep, pool, weather)
    print("Training rolling folds to collect val residuals...", flush=True)
    res = collect_val_residuals(dep, feats)

    pre = load_predeparture(CSV)
    merged = res.merge(pre, on=["airport", "date"], how="left")
    matched = merged["dly_per_dep"].notna()
    print(f"\n=== Coverage ===")
    print(f"  val flights: {len(merged):,}   joined to a delay airport-day: "
          f"{matched.sum():,} ({100*matched.mean():.1f}%)")

    m = merged[matched].copy()
    total_sse = (m["resid"] ** 2).sum()

    print(f"\n=== Error by daily pre-departure delay intensity (min/dep), {N_BUCKETS} buckets ===")
    print(f"{'bucket':>6} {'dly/dep':>16} {'n':>8} {'RMSE':>8} {'mean_resid':>11} {'%ofSSE':>8}")
    m["bucket"] = pd.qcut(m["dly_per_dep"], N_BUCKETS, labels=False, duplicates="drop")
    for b, g in m.groupby("bucket"):
        rmse = root_mean_squared_error(g["y"], g["pred"])
        share = 100 * (g["resid"] ** 2).sum() / total_sse
        lo, hi = g["dly_per_dep"].min(), g["dly_per_dep"].max()
        print(f"{int(b):>6} {lo:>7.1f}-{hi:<7.1f} {len(g):>8} {rmse:>8.1f} "
              f"{g['resid'].mean():>11.1f} {share:>7.1f}%")

    print(f"\n=== Decisive test: residual vs daily delay (does it explain what the model misses?) ===")
    r_signed = np.corrcoef(m["dly_per_dep"], m["resid"])[0, 1]
    r_abs = np.corrcoef(m["dly_per_dep"], m["resid"].abs())[0, 1]
    print(f"  corr(dly_per_dep, signed resid) = {r_signed:+.3f}  "
          f"(negative => we underpredict more as delay rises)")
    print(f"  corr(dly_per_dep, |resid|)      = {r_abs:+.3f}  "
          f"(positive => error magnitude grows with delay)")

    print(f"\n=== Per-airport (RMSE and mean signed resid on top-decile-delay days) ===")
    top = m[m["bucket"] == m["bucket"].max()]
    print(f"{'airport':>8} {'n_top':>7} {'RMSE_top':>9} {'meanResid_top':>14} {'RMSE_all':>9}")
    for apt in sorted(m["airport"].unique()):
        ga, gt = m[m["airport"] == apt], top[top["airport"] == apt]
        if len(gt) < 20:
            continue
        print(f"{apt:>8} {len(gt):>7} {root_mean_squared_error(gt['y'], gt['pred']):>9.1f} "
              f"{gt['resid'].mean():>14.1f} {root_mean_squared_error(ga['y'], ga['pred']):>9.1f}")


if __name__ == "__main__":
    main()

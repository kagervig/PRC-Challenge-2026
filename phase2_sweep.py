"""Phase 2: active_departures_queue variants.

The shipped queue feature (|{AOBT_j<=t}| - |{MVT_j<=t}| per airport, at t=AOBT_i) is the
crudest possible form. This tests four richer variants, each ADDED on top of the current
v30 feature set (which already includes the raw queue), so the delta measures marginal
value against the 315.02 summer-weighted anchor:

  1. runway_queue        — queue for the flight's specific runway, not the whole airport
  2. queue_norm          — raw queue / that airport's mean queue (cross-airport scaling)
  3. queue_rate_15m      — queue(t) - queue(t-15min): is the queue building or draining?
  4. queue_strict_ahead  — |{AOBT_j<t}| - |{MVT_j<=t}| (strict "ahead of me", excludes self)

Checkpointed + resumable to phase2_results.json (dir via SWEEP_OUT env var).
Variants live here, not model.py — they get promoted only if they pass the gate.
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

import model
import tune_harness as th

EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
OUT = Path(os.environ.get("SWEEP_OUT", ".")) / "phase2_results.json"


def _secs(series):
    return (series - EPOCH).dt.total_seconds()


def _queue_at(aobt_sorted, mvt_sorted, t):
    """pushed-back minus airborne at times t, from an airport/runway's sorted pools."""
    pushed = np.searchsorted(aobt_sorted, t, side="right")
    airborne = np.searchsorted(mvt_sorted, t, side="right")
    return (pushed - airborne).astype(float)


def compute_queue_grouped(dep, group_cols):
    """Queue length grouped by group_cols (e.g. airport, or airport+runway)."""
    result = pd.Series(np.nan, index=dep.index, dtype=float)
    ref_s = _secs(dep["AOBT_3_flt"].fillna(dep["MVT_TIME_UTC_mvt"]))
    aobt_s = _secs(dep["AOBT_3_flt"])
    mvt_s = _secs(dep["MVT_TIME_UTC_mvt"])
    for _, grp in dep.groupby(group_cols):
        idx = grp.index
        has = aobt_s.loc[idx].notna().values
        aobt = np.sort(aobt_s.loc[idx].values[has])
        mvt = np.sort(mvt_s.loc[idx].values[has])
        result.loc[idx] = _queue_at(aobt, mvt, ref_s.loc[idx].values)
    return result


def compute_queue_rate(dep, lag_minutes=15):
    """queue(t) - queue(t - lag): net change in queue over the last lag minutes."""
    lag_s = lag_minutes * 60.0
    result = pd.Series(np.nan, index=dep.index, dtype=float)
    ref_s = _secs(dep["AOBT_3_flt"].fillna(dep["MVT_TIME_UTC_mvt"]))
    aobt_s = _secs(dep["AOBT_3_flt"])
    mvt_s = _secs(dep["MVT_TIME_UTC_mvt"])
    for _, grp in dep.groupby("ADEP_mvt"):
        idx = grp.index
        has = aobt_s.loc[idx].notna().values
        aobt = np.sort(aobt_s.loc[idx].values[has])
        mvt = np.sort(mvt_s.loc[idx].values[has])
        t = ref_s.loc[idx].values
        result.loc[idx] = _queue_at(aobt, mvt, t) - _queue_at(aobt, mvt, t - lag_s)
    return result


def compute_queue_strict(dep):
    """Strict 'ahead of me': |{AOBT_j < t}| - |{MVT_j <= t}| (excludes self)."""
    result = pd.Series(np.nan, index=dep.index, dtype=float)
    ref_s = _secs(dep["AOBT_3_flt"].fillna(dep["MVT_TIME_UTC_mvt"]))
    aobt_s = _secs(dep["AOBT_3_flt"])
    mvt_s = _secs(dep["MVT_TIME_UTC_mvt"])
    for _, grp in dep.groupby("ADEP_mvt"):
        idx = grp.index
        has = aobt_s.loc[idx].notna().values
        aobt = np.sort(aobt_s.loc[idx].values[has])
        mvt = np.sort(mvt_s.loc[idx].values[has])
        t = ref_s.loc[idx].values
        pushed = np.searchsorted(aobt, t, side="left")   # strict <
        airborne = np.searchsorted(mvt, t, side="right")
        result.loc[idx] = (pushed - airborne).astype(float)
    return result


def load_done():
    return json.loads(OUT.read_text()) if OUT.exists() else {}


def main():
    print("Loading data + baseline matrix (current v30, incl. raw queue)...", flush=True)
    dep, pool, weather = th.load()
    feats = th.prepare(dep, pool, weather)
    done = load_done()

    # Establish the anchor once (current feature set, no variant) if not already recorded.
    if "baseline" not in done:
        h, c = th.evaluate(feats, dep, "BASELINE (v30, raw queue only)")
        done["baseline"] = [h, c]
        OUT.write_text(json.dumps(done, indent=2))
    base_h = done["baseline"][0]

    # queue_norm scales the raw queue by that airport's mean queue (cross-airport comparability).
    airport_mean_q = feats["active_departures_queue"].groupby(dep["ADEP_mvt"]).transform("mean")

    variants = {
        "runway_queue": lambda: compute_queue_grouped(dep, ["ADEP_mvt", "RUNWAY_mvt"]),
        "queue_norm": lambda: feats["active_departures_queue"] / airport_mean_q.replace(0, np.nan),
        "queue_rate_15m": lambda: compute_queue_rate(dep, 15),
        "queue_strict_ahead": lambda: compute_queue_strict(dep),
    }

    for name, compute in variants.items():
        if name in done:
            print(f"skip {name} (done: honest {done[name][0]:.2f})", flush=True)
            continue
        col = compute()
        print(f"\n{name} stats: min={col.min():.2f} med={col.median():.2f} "
              f"max={col.max():.2f} nan%={100*col.isna().mean():.1f}", flush=True)
        feats[name] = col
        h, c = th.evaluate(feats, dep, f"+ {name}")
        done[name] = [h, c]
        OUT.write_text(json.dumps(done, indent=2))
        del feats[name]  # remove before testing the next variant (marginal, not cumulative)

    print("\n\n==================== PHASE 2 SUMMARY ====================", flush=True)
    print(f"Baseline (raw queue only): honest {base_h:.2f}\n")
    for name in variants:
        if name not in done:
            continue
        h = done[name][0]
        tag = "IMPROVES" if h < base_h - 0.01 else "no gain"
        print(f"+ {name:20s} honest={h:.2f}  delta={h-base_h:+.2f}  [{tag}]")
    print(f"\nResults saved to {OUT}", flush=True)


if __name__ == "__main__":
    main()

"""Phase 5: gate-hold backlog — the cumulative undischarged pushback queue.

active_departures_queue counts flights that have pushed but not yet taken off (a taxiway
queue). It misses the pool that ATC gate-holding inflates: flights whose planned pushback
(EOBT) has passed but which have NOT actually pushed back yet. During a collapse, takeoff
throughput freezes and completions stop moving, but this overdue-at-the-gate backlog explodes.

Signal, at each flight's pushback time t = AOBT_3 (fallback MVT_TIME), same airport:
    overdue(t) = #{ EOBT_j <= t  AND  AOBT_j > t }
Each other flight j contributes an overdue interval [EOBT_j, AOBT_j): planned-push passed,
actual-push still ahead. This is an interval-stabbing count, so for well-formed intervals
(EOBT_j <= AOBT_j) it reduces to two strictly-backward cumulative counts at t:
    overdue(t) = #{EOBT_j <= t} - #{AOBT_j <= t}
Both terms only need events observed at or before t (how many EOBTs have passed, how many
flights have actually pushed by now); the future AOBT value itself is never read, only its
count <= t. So the signal is causal. EOBT_1 and AOBT_3 are both present in the ranking set,
so it is train/serve-consistent.

Guards:
  - Early pushes (AOBT_j < EOBT_j) form no overdue interval and are dropped — correct, they
    were never overdue.
  - Flights that never push (AOBT null: cancelled / date-rollover artifacts) are dropped, so
    the day-apart AOBT artifacts do not register as forever-overdue.
  - A capped variant additionally drops intervals longer than MAX_OVERDUE, bounding any
    residual artifact and matching the "overdue in the last 60-120 min" framing.

Tested added on top of the v33 baseline against the summer-weighted harness. Resumable.
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

import model
import tune_harness as th

EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
OUT = Path(os.environ.get("SWEEP_OUT", ".")) / "phase5_results.json"
MAX_OVERDUE_SEC = 2 * 3600.0


def _secs(s):
    return (s - EPOCH).dt.total_seconds()


def _overdue_grouped(dep, group_cols, max_overdue=None):
    """Overdue-pushback backlog at each flight's pushback, grouped by group_cols.

    max_overdue: if set, drop overdue intervals longer than this (seconds).
    """
    ref_s = _secs(dep["AOBT_3_flt"].fillna(dep["MVT_TIME_UTC_mvt"]))
    eobt_s = _secs(dep["EOBT_1_flt"])
    aobt_s = _secs(dep["AOBT_3_flt"])
    result = pd.Series(np.nan, index=dep.index, dtype=float)

    for _, grp in dep.groupby(group_cols):
        idx = grp.index
        eobt = eobt_s.loc[idx].values
        aobt = aobt_s.loc[idx].values
        valid = ~np.isnan(eobt) & ~np.isnan(aobt) & (eobt <= aobt)
        if max_overdue is not None:
            valid &= (aobt - eobt) <= max_overdue
        starts = np.sort(eobt[valid])
        ends = np.sort(aobt[valid])
        t = ref_s.loc[idx].values
        overdue = (np.searchsorted(starts, t, side="right")
                   - np.searchsorted(ends, t, side="right"))
        result.loc[idx] = overdue.astype(float)
    return result


def compute_signals(dep):
    return {
        "overdue_queue_2h": _overdue_grouped(dep, ["ADEP_mvt"], MAX_OVERDUE_SEC),
        "overdue_queue_uncapped": _overdue_grouped(dep, ["ADEP_mvt"], None),
        "overdue_runway_2h": _overdue_grouped(dep, ["ADEP_mvt", "RUNWAY_mvt"], MAX_OVERDUE_SEC),
    }


def load_done():
    return json.loads(OUT.read_text()) if OUT.exists() else {}


def main():
    print("Loading data + v33 baseline matrix...", flush=True)
    dep, pool, weather = th.load()
    feats = th.prepare(dep, pool, weather)
    sigs = compute_signals(dep)
    for name, col in sigs.items():
        print(f"  {name}: min={col.min():.2f} med={col.median():.2f} "
              f"mean={col.mean():.2f} max={col.max():.2f} nan%={100*col.isna().mean():.1f}",
              flush=True)

    done = load_done()
    if "baseline" not in done:
        h, c = th.evaluate(feats, dep, "BASELINE (v33)")
        done["baseline"] = [h, c]
        OUT.write_text(json.dumps(done, indent=2))
    base_h = done["baseline"][0]

    tests = {name: [name] for name in sigs}
    tests["all_three"] = list(sigs)

    for name, cols in tests.items():
        if name in done:
            print(f"skip {name} (honest {done[name][0]:.2f})", flush=True)
            continue
        for c in cols:
            feats[c] = sigs[c]
        h, cl = th.evaluate(feats, dep, f"+ {name}")
        done[name] = [h, cl]
        OUT.write_text(json.dumps(done, indent=2))
        for c in cols:
            del feats[c]

    print("\n\n==================== PHASE 5 SUMMARY ====================", flush=True)
    print(f"Baseline (v33): honest {base_h:.2f}\n")
    for name in list(tests):
        if name not in done:
            continue
        h = done[name][0]
        tag = "IMPROVES" if h < base_h - 0.01 else "no gain"
        print(f"+ {name:24s} honest={h:.2f}  delta={h-base_h:+.2f}  [{tag}]")
    print(f"\nResults saved to {OUT}", flush=True)


if __name__ == "__main__":
    main()

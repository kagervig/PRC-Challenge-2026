"""Phase 4: runway-reconfiguration signals — target LIRF-style config-change congestion.

LIRF taxi triples when the airport reconfigures (runway 25 -> 16R/34L, single-runway ops).
RUNWAY_mvt is a static per-flight feature; the model misses the *temporal* reconfiguration
state. These signals capture it, all causal (recent takeoffs before the flight's pushback) and
serve-available (RUNWAY_mvt is 0% null in the ranking set):

  active_runway_count_30m / _60m — distinct departure runways used in the last 30/60min
                                    (single- vs multi-runway ops = capacity state)
  minority_runway_share_60m      — fraction of recent same-airport takeoffs NOT on this
                                    flight's runway (high = on the off-nominal runway now)

Runway usage is timestamped at takeoff (MVT_TIME); each is queried at the flight's pushback
(AOBT_3). Each candidate is tested added on top of the v32 feature set, plus all three together,
against the summer-weighted harness. Checkpointed/resumable to phase4_results.json.
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

import model
import tune_harness as th

EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
OUT = Path(os.environ.get("SWEEP_OUT", ".")) / "phase4_results.json"


def _secs(s):
    return (s - EPOCH).dt.total_seconds()


def compute_runway_recon(dep):
    """Return dict of the three reconfiguration signals (see module docstring)."""
    ref_s = _secs(dep["AOBT_3_flt"].fillna(dep["MVT_TIME_UTC_mvt"]))
    mvt_s = _secs(dep["MVT_TIME_UTC_mvt"])
    arc30 = pd.Series(np.nan, index=dep.index, dtype=float)
    arc60 = pd.Series(np.nan, index=dep.index, dtype=float)
    minshare = pd.Series(np.nan, index=dep.index, dtype=float)

    for _, grp in dep.groupby("ADEP_mvt"):
        idx = grp.index
        t = ref_s.loc[idx].values
        ev_t = mvt_s.loc[idx].values                       # takeoff times = runway-use events
        rwy = grp["RUNWAY_mvt"].fillna("UNK").values
        ev_sorted = np.sort(ev_t)
        tot60 = (np.searchsorted(ev_sorted, t, side="left")
                 - np.searchsorted(ev_sorted, t - 3600, side="left"))
        d30 = np.zeros(len(idx)); d60 = np.zeros(len(idx)); own60 = np.zeros(len(idx))
        for r in pd.unique(rwy):
            rt = np.sort(ev_t[rwy == r])
            c30 = (np.searchsorted(rt, t, side="left")
                   - np.searchsorted(rt, t - 1800, side="left"))
            c60 = (np.searchsorted(rt, t, side="left")
                   - np.searchsorted(rt, t - 3600, side="left"))
            d30 += (c30 > 0); d60 += (c60 > 0)
            own60 += np.where(rwy == r, c60, 0)
        arc30.loc[idx] = d30
        arc60.loc[idx] = d60
        minshare.loc[idx] = np.where(tot60 > 0, 1 - own60 / np.maximum(tot60, 1), np.nan)

    return {"active_runway_count_30m": arc30,
            "active_runway_count_60m": arc60,
            "minority_runway_share_60m": minshare}


def load_done():
    return json.loads(OUT.read_text()) if OUT.exists() else {}


def main():
    print("Loading data + v32 baseline matrix...", flush=True)
    dep, pool, weather = th.load()
    feats = th.prepare(dep, pool, weather)
    recon = compute_runway_recon(dep)
    for name, col in recon.items():
        print(f"  {name}: min={col.min():.2f} med={col.median():.2f} "
              f"max={col.max():.2f} nan%={100*col.isna().mean():.1f}", flush=True)

    done = load_done()
    if "baseline" not in done:
        h, c = th.evaluate(feats, dep, "BASELINE (v32)")
        done["baseline"] = [h, c]
        OUT.write_text(json.dumps(done, indent=2))
    base_h = done["baseline"][0]

    tests = {name: [name] for name in recon}
    tests["all_three"] = list(recon)

    for name, cols in tests.items():
        if name in done:
            print(f"skip {name} (honest {done[name][0]:.2f})", flush=True)
            continue
        for c in cols:
            feats[c] = recon[c]
        h, cl = th.evaluate(feats, dep, f"+ {name}")
        done[name] = [h, cl]
        OUT.write_text(json.dumps(done, indent=2))
        for c in cols:
            del feats[c]

    print("\n\n==================== PHASE 4 SUMMARY ====================", flush=True)
    print(f"Baseline (v32): honest {base_h:.2f}\n")
    for name in list(tests):
        if name not in done:
            continue
        h = done[name][0]
        tag = "IMPROVES" if h < base_h - 0.01 else "no gain"
        print(f"+ {name:26s} honest={h:.2f}  delta={h-base_h:+.2f}  [{tag}]")
    print(f"\nResults saved to {OUT}", flush=True)


if __name__ == "__main__":
    main()

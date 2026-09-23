"""Phase 3: EWMA (exponential decay) vs boxcar windows for congestion_signal & recent_delay.

The boxcar window is a hard cutoff — a flight 59 min ago counts fully, 61 min ago not at all.
EWMA weights past events by exp(-age/tau), fading smoothly. This sweeps the half-life for an
EWMA version of each feature and compares it to the boxcar baseline.

Set CONG_BOXCAR_WINDOW / RECENT_BOXCAR_WINDOW to the Phase 1 winning windows once known so
EWMA is judged against the *tuned* boxcar, not the original 60-min one. Defaults are 60.

Numerically stable + vectorised: causal EWMA over irregular timestamps is done with
pandas ewm(halflife=, times=) on the event pool. The EWMA mean at a query time equals its
value at the last pool event strictly before it (the decay factor cancels in the weighted-mean
ratio), so each departure is mapped to that event via searchsorted — no per-row Python loop
over time, no exp() overflow.

Checkpointed + resumable to phase3_results.json (dir via SWEEP_OUT env var).
EWMA feature defs live here; promote a winner into model.py only if it passes the gate.
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

import model
import tune_harness as th

# Set these to the Phase 1 winners once phase1_results.json is in; defaults reproduce v30.
CONG_BOXCAR_WINDOW = 60
RECENT_BOXCAR_WINDOW = 60

HALFLIVES = [10, 20, 30, 45, 60, 90]
OUT = Path(os.environ.get("SWEEP_OUT", ".")) / "phase3_results.json"


def _ewma_pool_query(pool_times, pool_vals, query_times, halflife_min):
    """
    Causal EWMA of (pool_times, pool_vals) evaluated at query_times.

    Returns, for each query time t, the exp-decay-weighted mean of pool values whose event
    time is strictly < t, half-life = halflife_min. NaN where no prior event exists.
    """
    valid = ~np.isnan(pool_vals)
    pt = pool_times[valid]
    pv = pool_vals[valid]
    if pt.size == 0:
        return np.full(query_times.shape, np.nan)
    order = np.argsort(pt, kind="stable")
    pt = pt[order]
    pv = pv[order]
    ewma = (
        pd.Series(pv)
        .ewm(halflife=pd.Timedelta(minutes=halflife_min), times=pd.DatetimeIndex(pt))
        .mean()
        .to_numpy()
    )
    # last pool event strictly before each query time
    k = np.searchsorted(pt, query_times, side="left") - 1
    out = np.where(k >= 0, ewma[np.clip(k, 0, len(ewma) - 1)], np.nan)
    return out


def compute_congestion_ewma(dep, pool, halflife_min):
    """EWMA arrival taxi-in at the same airport, queried at each departure's pushback."""
    result = pd.Series(np.nan, index=dep.index, dtype=float)
    pool_t = pool["MVT_TIME_UTC_mvt"].to_numpy()
    pool_v = pool["TAXITIME_SEC_mvt"].to_numpy(dtype=float)
    ref = dep["AOBT_3_flt"].fillna(dep["MVT_TIME_UTC_mvt"])
    pool_ap = pool["ADEP_mvt"].to_numpy()
    for airport, grp in dep.groupby("ADEP_mvt"):
        sel = pool_ap == airport
        if not sel.any():
            continue
        q = ref.loc[grp.index].to_numpy()
        result.loc[grp.index] = _ewma_pool_query(pool_t[sel], pool_v[sel], q, halflife_min)
    return result


def compute_recent_delay_ewma(dep, halflife_min):
    """EWMA off-block delay (AOBT-SCHED) of prior same-airport departures."""
    result = pd.Series(np.nan, index=dep.index, dtype=float)
    ref = dep["AOBT_3_flt"].fillna(dep["MVT_TIME_UTC_mvt"])
    delay = (ref - dep["SCHED_TIME_UTC_mvt"]).dt.total_seconds()
    ref_np = ref.to_numpy()
    for _, grp in dep.groupby("ADEP_mvt"):
        idx = grp.index
        t = ref_np[dep.index.get_indexer(idx)]
        # pool == the departures themselves; searchsorted side='left'-1 excludes self
        result.loc[idx] = _ewma_pool_query(t, delay.loc[idx].to_numpy(), t, halflife_min)
    return result


def load_done():
    return json.loads(OUT.read_text()) if OUT.exists() else {}


def main():
    print(f"Loading data + boxcar baseline (cong={CONG_BOXCAR_WINDOW}, "
          f"recent={RECENT_BOXCAR_WINDOW})...", flush=True)
    dep, pool, weather = th.load()
    feats = th.prepare(dep, pool, weather,
                       cong_window=CONG_BOXCAR_WINDOW, recent_window=RECENT_BOXCAR_WINDOW)
    base_cong = feats["congestion_signal"].copy()
    base_recent = feats["recent_delay"].copy()
    done = load_done()

    if "baseline" not in done:
        h, c = th.evaluate(feats, dep, "BASELINE (boxcar)")
        done["baseline"] = [h, c]
        OUT.write_text(json.dumps(done, indent=2))
    base_h = done["baseline"][0]

    sweeps = {
        "congestion_signal": (base_cong, lambda hl: compute_congestion_ewma(dep, pool, hl)),
        "recent_delay": (base_recent, lambda hl: compute_recent_delay_ewma(dep, hl)),
    }

    for feature, (base_col, recompute) in sweeps.items():
        print(f"\n########## EWMA SWEEP: {feature} ##########", flush=True)
        for hl in HALFLIVES:
            key = f"{feature}|hl{hl}"
            if key in done:
                print(f"  skip {key} (honest {done[key][0]:.2f})", flush=True)
                continue
            col = recompute(hl)
            print(f"  {key} stats: med={col.median():.2f} nan%={100*col.isna().mean():.1f}",
                  flush=True)
            feats[feature] = col
            h, c = th.evaluate(feats, dep, f"{feature} EWMA hl={hl}min")
            done[key] = [h, c]
            OUT.write_text(json.dumps(done, indent=2))
        feats[feature] = base_col  # restore boxcar before next feature

    print("\n\n==================== PHASE 3 SUMMARY ====================", flush=True)
    print(f"Boxcar baseline: honest {base_h:.2f}\n")
    for feature in sweeps:
        rows = [(hl, done[f"{feature}|hl{hl}"][0]) for hl in HALFLIVES
                if f"{feature}|hl{hl}" in done]
        if not rows:
            continue
        best = min(rows, key=lambda r: r[1])
        tag = "IMPROVES" if best[1] < base_h - 0.01 else "no gain vs boxcar"
        print(f"{feature}: best EWMA hl={best[0]}min honest={best[1]:.2f}  "
              f"vs boxcar {base_h:.2f}  delta={best[1]-base_h:+.2f}  [{tag}]")
        print("   " + "  ".join(f"hl{hl}:{h:.1f}" for hl, h in rows))
    print(f"\nResults saved to {OUT}", flush=True)


if __name__ == "__main__":
    main()

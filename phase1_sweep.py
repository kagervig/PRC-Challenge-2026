"""Phase 1: decouple the shared 60-min window and sweep each feature independently.

congestion_signal, recent_delay, and arrival_demand currently all share
CONGESTION_WINDOW_MINUTES=60. Sweep each over WINDOWS holding the other two at 60,
measured on the summer-weighted harness against the 315.02 pooled-honest anchor.

Efficient: builds the baseline matrix once, then recomputes only the swept feature's
column in place, restoring it to the 60-min value before moving to the next feature.

Checkpointed + resumable: each finished condition is appended to phase1_results.json
(dir overridable via SWEEP_OUT env var), so a disconnect or kill never loses progress —
re-running skips conditions already recorded.
"""
import json
import os
from pathlib import Path

import model
import tune_harness as th

WINDOWS = [15, 30, 45, 60, 90, 120]
OUT = Path(os.environ.get("SWEEP_OUT", ".")) / "phase1_results.json"


def load_done():
    if OUT.exists():
        return json.loads(OUT.read_text())
    return {}


def save(done):
    OUT.write_text(json.dumps(done, indent=2))


print("Loading data + baseline matrix (all windows = 60)...", flush=True)
dep, pool, weather = th.load()
feats = th.prepare(dep, pool, weather)  # all three windows at 60

base_cols = {
    "congestion_signal": feats["congestion_signal"].copy(),
    "recent_delay": feats["recent_delay"].copy(),
    "arrival_demand": feats["arrival_demand"].copy(),
}

done = load_done()  # {"feature|window": [honest, clean]}


def sweep(feature, recompute):
    print(f"\n########## SWEEP: {feature} ##########", flush=True)
    for w in WINDOWS:
        key = f"{feature}|{w}"
        if key in done:
            print(f"  skip {key} (done: honest {done[key][0]:.2f})", flush=True)
            continue
        feats[feature] = base_cols[feature] if w == 60 else recompute(w)
        h, c = th.evaluate(feats, dep, f"{feature} window={w}min")
        done[key] = [h, c]
        save(done)  # checkpoint after every condition
    feats[feature] = base_cols[feature]  # restore to 60 before next feature


sweep("congestion_signal", lambda w: model.compute_congestion_signal(dep, pool, w))
sweep("recent_delay", lambda w: model.compute_recent_delay(dep, w))
sweep("arrival_demand", lambda w: model.compute_arrival_demand(dep, pool, w))

print("\n\n==================== PHASE 1 SUMMARY ====================", flush=True)
for feature in ["congestion_signal", "recent_delay", "arrival_demand"]:
    rows = [(w, done[f"{feature}|{w}"][0]) for w in WINDOWS if f"{feature}|{w}" in done]
    if not rows:
        continue
    b60 = dict(rows).get(60)
    best = min(rows, key=lambda r: r[1])
    tag = "IMPROVES" if b60 and best[1] < b60 - 0.01 else "no gain"
    base_str = f"{b60:.2f}" if b60 else "n/a"
    print(f"{feature}: best={best[0]}min honest={best[1]:.2f}  vs 60min={base_str}  [{tag}]")
    print("   " + "  ".join(f"{w}:{h:.1f}" for w, h in rows))
print(f"\nResults saved to {OUT}", flush=True)

"""
Targeted significance analysis for three open questions:
  1. Month of year — per airport (11 separate KW tests)
  2. Operating airline — overall KW
  3. Specific gate/stand — per airport KW

Uses Kruskal-Wallis with epsilon-squared effect size.
  ε² < 0.01: negligible  |  0.01–0.04: small  |  0.04–0.14: medium  |  >0.14: large
"""

import glob
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

TARGET = "TAXITIME_SEC_mvt"
MIN_GROUP_N = 30  # minimum observations per group to include in tests


def epsilon_squared(h_stat: float, n: int, k: int) -> float:
    return (h_stat - k + 1) / (n - k)


def classify_effect(eps2: float) -> str:
    if eps2 < 0.01:
        return "negligible"
    elif eps2 < 0.04:
        return "small"
    elif eps2 < 0.14:
        return "medium"
    return "large"


def kw_test(series_by_group: dict) -> dict | None:
    """Run KW on a dict of {group_label: array}. Filters groups below MIN_GROUP_N."""
    filtered = {k: v for k, v in series_by_group.items() if len(v) >= MIN_GROUP_N}
    if len(filtered) < 2:
        return None
    groups = list(filtered.values())
    h, p = stats.kruskal(*groups)
    n = sum(len(g) for g in groups)
    k = len(groups)
    eps2 = epsilon_squared(h, n, k)
    return {"n_groups": k, "n_obs": n, "h_stat": h, "p_value": p, "effect_size": eps2,
            "effect_label": classify_effect(eps2)}


def load_data() -> pd.DataFrame:
    files = sorted(glob.glob("training_*.parquet"))
    print(f"Loading {len(files)} training files...")
    frames = []
    for f in files:
        frames.append(pd.read_parquet(f, columns=[
            TARGET, "MVT_TIME_UTC_mvt", "PHASE_mvt",
            "ADEP_mvt", "ADES_mvt",
            "AIRCRAFT_OPERATOR_flt",
            "STAND_mvt",
        ]))
    df = pd.concat(frames, ignore_index=True)
    df = df[df[TARGET] > 0]
    print(f"  Total rows: {len(df):,}")
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df["airport"] = df.apply(
        lambda r: r["ADEP_mvt"] if r["PHASE_mvt"] == "DEP" else r["ADES_mvt"], axis=1
    )
    df["month"] = df["MVT_TIME_UTC_mvt"].dt.month
    return df


# ── Q1: Month of year, per airport ───────────────────────────────────────────

def q1_month_per_airport(df: pd.DataFrame):
    print("\n" + "=" * 70)
    print("Q1 — Month of year effect on taxi time, per airport")
    print("=" * 70)
    print(f"{'Airport':<8} {'Months':>6} {'N':>9} {'p-value':>12} {'ε²':>8} {'Magnitude'}")
    print("-" * 70)

    results = []
    for airport, adf in df.groupby("airport"):
        groups = {m: g[TARGET].values for m, g in adf.groupby("month")}
        r = kw_test(groups)
        if r is None:
            continue
        r["airport"] = airport
        results.append(r)

    results.sort(key=lambda x: x["effect_size"], reverse=True)
    for r in results:
        p = r["p_value"]
        p_str = f"{p:.2e}" if p < 0.001 else f"{p:.4f}"
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        print(f"{r['airport']:<8} {r['n_groups']:>6} {r['n_obs']:>9} {p_str:>12} {r['effect_size']:>8.4f}  {r['effect_label']:<10} {sig}")

    print("\nMonthly median taxi times per airport (minutes):")
    pivot = (
        df.groupby(["airport", "month"])[TARGET]
        .median()
        .div(60)
        .round(1)
        .unstack(level="month")
    )
    pivot.columns = [f"M{c:02d}" for c in pivot.columns]
    pivot["range"] = pivot.max(axis=1) - pivot.min(axis=1)
    pivot = pivot.sort_values("range", ascending=False)
    print(pivot.to_string())


# ── Q2: Operating airline ─────────────────────────────────────────────────────

def q2_airline(df: pd.DataFrame):
    print("\n" + "=" * 70)
    print("Q2 — Operating airline effect on taxi time (overall)")
    print("=" * 70)

    groups = {op: g[TARGET].values for op, g in df.groupby("AIRCRAFT_OPERATOR_flt")}
    r = kw_test(groups)
    if r is None:
        print("  Not enough data.")
        return

    p = r["p_value"]
    p_str = f"{p:.2e}" if p < 0.001 else f"{p:.4f}"
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
    print(f"  Airlines tested: {r['n_groups']}  |  N: {r['n_obs']:,}  |  p: {p_str} {sig}  |  ε² = {r['effect_size']:.4f}  ({r['effect_label']})")

    # Top and bottom airlines by median (min sample size for reliability)
    airline_stats = (
        df.groupby("AIRCRAFT_OPERATOR_flt")[TARGET]
        .agg(median="median", n="count")
    )
    airline_stats = airline_stats[airline_stats["n"] >= 500]
    airline_stats["median_min"] = (airline_stats["median"] / 60).round(1)
    airline_stats = airline_stats.sort_values("median")

    print(f"\n  Fastest 5 airlines (median taxi, min n=500):")
    for op, row in airline_stats.head(5).iterrows():
        print(f"    {op[:20]:<20}  {row['median_min']:>5.1f} min  (n={row['n']:,})")

    print(f"\n  Slowest 5 airlines (median taxi, min n=500):")
    for op, row in airline_stats.tail(5).iterrows():
        print(f"    {op[:20]:<20}  {row['median_min']:>5.1f} min  (n={row['n']:,})")


# ── Q3: Specific stand/gate, per airport ─────────────────────────────────────

def q3_stand_per_airport(df: pd.DataFrame):
    print("\n" + "=" * 70)
    print("Q3 — Stand/gate effect on taxi time, per airport")
    print("=" * 70)
    print(f"{'Airport':<8} {'Stands':>7} {'N':>9} {'p-value':>12} {'ε²':>8} {'Magnitude'}")
    print("-" * 70)

    results = []
    for airport, adf in df.groupby("airport"):
        groups = {s: g[TARGET].values for s, g in adf.groupby("STAND_mvt")}
        r = kw_test(groups)
        if r is None:
            continue
        r["airport"] = airport
        r["adf"] = adf  # keep for follow-up
        results.append(r)

    results.sort(key=lambda x: x["effect_size"], reverse=True)
    for r in results:
        p = r["p_value"]
        p_str = f"{p:.2e}" if p < 0.001 else f"{p:.4f}"
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        print(f"{r['airport']:<8} {r['n_groups']:>7} {r['n_obs']:>9} {p_str:>12} {r['effect_size']:>8.4f}  {r['effect_label']:<10} {sig}")

    # For the top airport by effect size, show the stand spread
    top = results[0]
    airport = top["airport"]
    adf = top["adf"]
    stand_stats = (
        adf.groupby("STAND_mvt")[TARGET]
        .agg(median="median", n="count")
    )
    stand_stats = stand_stats[stand_stats["n"] >= MIN_GROUP_N]
    stand_stats["median_min"] = (stand_stats["median"] / 60).round(1)
    stand_stats = stand_stats.sort_values("median")

    print(f"\n  Stand spread at {airport} (highest ε²), min n={MIN_GROUP_N}:")
    print(f"  Overall range: {stand_stats['median_min'].min()} – {stand_stats['median_min'].max()} min")
    print(f"\n  Fastest 5 stands:")
    print(stand_stats.head(5)[["median_min", "n"]].rename(columns={"median_min": "median (min)"}).to_string())
    print(f"\n  Slowest 5 stands:")
    print(stand_stats.tail(5)[["median_min", "n"]].rename(columns={"median_min": "median (min)"}).to_string())


def main():
    df = load_data()
    df = add_features(df)

    q1_month_per_airport(df)
    q2_airline(df)
    q3_stand_per_airport(df)


if __name__ == "__main__":
    main()

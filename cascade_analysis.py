"""
Cascade analysis: does a disruption at an airport create a knock-on effect
for subsequent flights?

Two approaches:
  1. Lag correlation — for each flight, how well does the rolling mean taxi
     time of the previous N minutes predict the current flight's taxi time?
     Measured across windows: 15, 30, 60, 90, 120, 180 minutes.

  2. Time series — hourly mean taxi time through a bad day vs a normal day
     at the same airport, to visualise onset and recovery.
"""

import glob
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

TARGET = "TAXITIME_SEC_mvt"
WINDOWS = ["15min", "30min", "60min", "90min", "120min", "180min"]

# Known bad days for time-series visualisation
BAD_DAYS = [
    ("LTFM", "2025-02-23"),  # worst day overall
    ("LIRF", "2025-07-13"),  # worst congestion day
    ("LSZH", "2025-08-12"),  # worst Zurich day
]


def load_data() -> pd.DataFrame:
    files = sorted(glob.glob("training_*.parquet"))
    print(f"Loading {len(files)} training files...")
    frames = []
    for f in files:
        frames.append(pd.read_parquet(f, columns=[
            TARGET, "MVT_TIME_UTC_mvt", "PHASE_mvt", "ADEP_mvt", "ADES_mvt",
        ]))
    df = pd.concat(frames, ignore_index=True)
    df = df[df[TARGET] > 0]
    df["airport"] = df.apply(
        lambda r: r["ADEP_mvt"] if r["PHASE_mvt"] == "DEP" else r["ADES_mvt"], axis=1
    )
    df = df.sort_values(["airport", "MVT_TIME_UTC_mvt"]).reset_index(drop=True)
    print(f"  Total rows: {len(df):,}")
    return df


def compute_deviation(df: pd.DataFrame) -> pd.DataFrame:
    airport_median = df.groupby("airport")[TARGET].median()
    df["baseline"] = df["airport"].map(airport_median)
    df["deviation"] = df[TARGET] - df["baseline"]
    return df


def compute_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """For each flight, compute the rolling mean deviation of preceding flights
    at the same airport over several time windows."""
    print("Computing rolling features (this takes a moment)...")
    parts = []
    for airport, adf in df.groupby("airport"):
        adf = adf.set_index("MVT_TIME_UTC_mvt").sort_index()
        for w in WINDOWS:
            # shift(1) excludes the current flight from its own window
            col = f"roll_{w}"
            adf[col] = (
                adf["deviation"]
                .rolling(w, min_periods=3)
                .mean()
                .shift(1)
            )
        adf = adf.reset_index()
        parts.append(adf)
    return pd.concat(parts, ignore_index=True)


def lag_correlation_analysis(df: pd.DataFrame):
    print("\n" + "=" * 65)
    print("LAG CORRELATION — rolling mean deviation vs current taxi deviation")
    print("(Spearman r, all airports combined)")
    print("=" * 65)
    print(f"  {'Window':<10} {'Spearman r':>12} {'p-value':>14} {'Strength'}")
    print("  " + "-" * 55)

    for w in WINDOWS:
        col = f"roll_{w}"
        valid = df[[col, "deviation"]].dropna()
        r, p = stats.spearmanr(valid[col], valid["deviation"])
        strength = (
            "strong" if abs(r) >= 0.5
            else "moderate" if abs(r) >= 0.3
            else "weak" if abs(r) >= 0.1
            else "negligible"
        )
        p_str = f"{p:.2e}" if p < 0.001 else f"{p:.4f}"
        print(f"  {w:<10} {r:>12.4f} {p_str:>14}  {strength}")


def per_airport_correlation(df: pd.DataFrame):
    """Best window (60min) broken down by airport."""
    col = "roll_60min"
    print("\n" + "=" * 65)
    print("60-MINUTE LAG CORRELATION — per airport")
    print("=" * 65)
    print(f"  {'Airport':<8} {'Spearman r':>12} {'N':>10}  {'Strength'}")
    print("  " + "-" * 50)

    results = []
    for airport, adf in df.groupby("airport"):
        valid = adf[[col, "deviation"]].dropna()
        if len(valid) < 100:
            continue
        r, p = stats.spearmanr(valid[col], valid["deviation"])
        results.append((airport, r, len(valid)))

    results.sort(key=lambda x: x[1], reverse=True)
    for airport, r, n in results:
        strength = (
            "strong" if abs(r) >= 0.5
            else "moderate" if abs(r) >= 0.3
            else "weak" if abs(r) >= 0.1
            else "negligible"
        )
        print(f"  {airport:<8} {r:>12.4f} {n:>10,}  {strength}")


def timeseries_bad_vs_normal(df: pd.DataFrame):
    print("\n" + "=" * 65)
    print("TIME SERIES — bad day vs typical day (hourly mean, minutes)")
    print("=" * 65)

    for airport, bad_date in BAD_DAYS:
        adf = df[df["airport"] == airport].copy()
        adf["date"] = adf["MVT_TIME_UTC_mvt"].dt.date.astype(str)
        adf["hour"] = adf["MVT_TIME_UTC_mvt"].dt.hour

        bad = adf[adf["date"] == bad_date]
        # "typical" = median hourly mean across all other days
        other = adf[adf["date"] != bad_date]
        typical_hourly = other.groupby(["date", "hour"])[TARGET].mean().groupby("hour").median() / 60

        bad_hourly = bad.groupby("hour")[TARGET].mean() / 60

        print(f"\n  {airport} — {bad_date} (bad) vs typical")
        print(f"  {'Hour':<6} {'Bad day':>9} {'Typical':>9} {'Excess':>9}")
        print("  " + "-" * 38)
        for hour in range(24):
            b = bad_hourly.get(hour, np.nan)
            t = typical_hourly.get(hour, np.nan)
            if np.isnan(b) and np.isnan(t):
                continue
            excess = (b - t) if not (np.isnan(b) or np.isnan(t)) else np.nan
            b_str = f"{b:.1f}" if not np.isnan(b) else "  —"
            t_str = f"{t:.1f}" if not np.isnan(t) else "  —"
            e_str = f"+{excess:.1f}" if (not np.isnan(excess) and excess > 0) else (f"{excess:.1f}" if not np.isnan(excess) else "  —")
            print(f"  {hour:02d}:00  {b_str:>9} {t_str:>9} {e_str:>9}")


def main():
    df = load_data()
    df = compute_deviation(df)
    df = compute_rolling_features(df)

    lag_correlation_analysis(df)
    per_airport_correlation(df)
    timeseries_bad_vs_normal(df)


if __name__ == "__main__":
    main()

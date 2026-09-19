"""
Are some airlines reliably faster or slower than the airport baseline,
across multiple airports?

Approach:
  1. Compute each airport's median taxi time as a baseline.
  2. For each (airport, airline) cell, compute deviation from that baseline.
  3. For airlines present at enough airports, assess whether their
     deviation is consistent in sign (always slower, or always faster).
"""

import glob
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

TARGET = "TAXITIME_SEC_mvt"
MIN_CELL_N = 100    # minimum flights per (airport, airline) to include cell
MIN_AIRPORTS = 3    # minimum airports an airline must appear at to assess consistency


def load_data() -> pd.DataFrame:
    files = sorted(glob.glob("training_*.parquet"))
    print(f"Loading {len(files)} training files...")
    frames = []
    for f in files:
        frames.append(pd.read_parquet(f, columns=[
            TARGET, "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "AIRCRAFT_OPERATOR_flt",
        ]))
    df = pd.concat(frames, ignore_index=True)
    df = df[df[TARGET] > 0]
    df["airport"] = df.apply(
        lambda r: r["ADEP_mvt"] if r["PHASE_mvt"] == "DEP" else r["ADES_mvt"], axis=1
    )
    print(f"  Total rows: {len(df):,}")
    return df


def main():
    df = load_data()

    # Airport baselines
    airport_median = df.groupby("airport")[TARGET].median()

    # Per (airport, airline) stats
    cell_stats = (
        df.groupby(["airport", "AIRCRAFT_OPERATOR_flt"])[TARGET]
        .agg(median="median", n="count")
        .reset_index()
    )
    cell_stats = cell_stats[cell_stats["n"] >= MIN_CELL_N]

    # Deviation from airport baseline (minutes)
    cell_stats["baseline"] = cell_stats["airport"].map(airport_median)
    cell_stats["deviation_min"] = (cell_stats["median"] - cell_stats["baseline"]) / 60

    # Filter to airlines present at enough airports
    airports_per_airline = cell_stats.groupby("AIRCRAFT_OPERATOR_flt")["airport"].nunique()
    multi_airport_airlines = airports_per_airline[airports_per_airline >= MIN_AIRPORTS].index
    multi = cell_stats[cell_stats["AIRCRAFT_OPERATOR_flt"].isin(multi_airport_airlines)].copy()

    print(f"\nAirlines with ≥{MIN_AIRPORTS} airports and ≥{MIN_CELL_N} flights/airport: {len(multi_airport_airlines)}")

    # Summary per airline: mean deviation, consistency (fraction of airports where they're slower)
    summary = multi.groupby("AIRCRAFT_OPERATOR_flt").agg(
        n_airports=("airport", "nunique"),
        total_flights=("n", "sum"),
        mean_deviation_min=("deviation_min", "mean"),
        airports_slower=("deviation_min", lambda x: (x > 0).sum()),
        airports_faster=("deviation_min", lambda x: (x < 0).sum()),
    ).reset_index()

    summary["pct_slower"] = summary["airports_slower"] / summary["n_airports"]
    summary["consistent"] = summary["pct_slower"].apply(
        lambda p: "always slower" if p == 1.0
        else ("always faster" if p == 0.0
        else ("mostly slower" if p >= 0.75
        else ("mostly faster" if p <= 0.25
        else "mixed")))
    )
    summary = summary.sort_values("mean_deviation_min", ascending=False)

    print("\n" + "=" * 80)
    print("AIRLINE CONSISTENCY — deviation from airport baseline")
    print(f"(airlines at ≥{MIN_AIRPORTS} airports with ≥{MIN_CELL_N} flights each)")
    print("=" * 80)
    print(f"{'Airline':<22} {'Airports':>8} {'Flights':>9} {'Mean dev':>9} {'Pattern'}")
    print("-" * 80)
    for _, r in summary.iterrows():
        sign = "+" if r["mean_deviation_min"] > 0 else ""
        print(f"{r['AIRCRAFT_OPERATOR_flt'][:22]:<22} {int(r['n_airports']):>8} {int(r['total_flights']):>9} "
              f"  {sign}{r['mean_deviation_min']:>+.1f} min  {r['consistent']}")

    # Breakdown: consistently fast vs slow
    always_slower = summary[summary["consistent"] == "always slower"]
    always_faster = summary[summary["consistent"] == "always faster"]
    mostly_slower = summary[summary["consistent"].isin(["always slower", "mostly slower"])]
    mostly_faster = summary[summary["consistent"].isin(["always faster", "mostly faster"])]

    print(f"\nConsistently slower (always or mostly): {len(mostly_slower)} airlines")
    print(f"Consistently faster (always or mostly): {len(mostly_faster)} airlines")
    print(f"Mixed: {len(summary) - len(mostly_slower) - len(mostly_faster)} airlines")

    # Show per-airport breakdown for the most extreme consistent airlines
    n_show = 3
    print(f"\n{'='*80}")
    print(f"Per-airport breakdown for top {n_show} most reliably SLOW airlines")
    print(f"{'='*80}")
    for _, row in mostly_slower.head(n_show).iterrows():
        airline = row["AIRCRAFT_OPERATOR_flt"]
        adf = multi[multi["AIRCRAFT_OPERATOR_flt"] == airline][["airport", "deviation_min", "n"]].sort_values("deviation_min", ascending=False)
        print(f"\n{airline[:30]}  (mean dev: {row['mean_deviation_min']:+.1f} min)")
        print(adf.rename(columns={"deviation_min": "dev (min)"}).to_string(index=False))

    print(f"\n{'='*80}")
    print(f"Per-airport breakdown for top {n_show} most reliably FAST airlines")
    print(f"{'='*80}")
    for _, row in mostly_faster.tail(n_show).iterrows():
        airline = row["AIRCRAFT_OPERATOR_flt"]
        adf = multi[multi["AIRCRAFT_OPERATOR_flt"] == airline][["airport", "deviation_min", "n"]].sort_values("deviation_min")
        print(f"\n{airline[:30]}  (mean dev: {row['mean_deviation_min']:+.1f} min)")
        print(adf.rename(columns={"deviation_min": "dev (min)"}).to_string(index=False))


if __name__ == "__main__":
    main()

"""
Statistical significance analysis of factors affecting taxi time.

Uses Kruskal-Wallis (multi-group) and Mann-Whitney U (two-group) tests.
Effect size: epsilon-squared (ε²) — comparable across different numbers of groups.
  ε² < 0.01: negligible
  ε² 0.01–0.04: small
  ε² 0.04–0.14: medium
  ε² > 0.14: large
"""

import glob
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")


def epsilon_squared(h_stat: float, n: int, k: int) -> float:
    """Epsilon-squared effect size for Kruskal-Wallis."""
    return (h_stat - k + 1) / (n - k)


def rank_biserial(u_stat: float, n1: int, n2: int) -> float:
    """Rank-biserial correlation effect size for Mann-Whitney U."""
    return 1 - (2 * u_stat) / (n1 * n2)


def kw_test(df: pd.DataFrame, col: str, target: str) -> dict:
    groups = [grp[target].dropna().values for _, grp in df.groupby(col) if len(grp) > 1]
    if len(groups) < 2:
        return None
    h, p = stats.kruskal(*groups)
    n = sum(len(g) for g in groups)
    k = len(groups)
    eps2 = epsilon_squared(h, n, k)
    return {"variable": col, "n_groups": k, "n_obs": n, "statistic": h, "p_value": p, "effect_size": eps2}


def mwu_test(df: pd.DataFrame, col: str, target: str) -> dict:
    groups = {name: grp[target].dropna().values for name, grp in df.groupby(col)}
    if len(groups) != 2:
        return None
    names = list(groups.keys())
    g1, g2 = groups[names[0]], groups[names[1]]
    u, p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
    r = abs(rank_biserial(u, len(g1), len(g2)))
    return {"variable": col, "n_groups": 2, "n_obs": len(g1) + len(g2), "statistic": u, "p_value": p, "effect_size": r}


def classify_effect(eps2: float) -> str:
    if eps2 < 0.01:
        return "negligible"
    elif eps2 < 0.04:
        return "small"
    elif eps2 < 0.14:
        return "medium"
    return "large"


def load_data() -> pd.DataFrame:
    files = sorted(glob.glob("training_*.parquet"))
    print(f"Loading {len(files)} training files...")
    frames = []
    for f in files:
        frames.append(pd.read_parquet(f, columns=[
            "TAXITIME_SEC_mvt", "MVT_TIME_UTC_mvt", "PHASE_mvt",
            "RUNWAY_mvt", "WK_TBL_CAT_flt", "MARKET_SEGMENT_flt",
            "FLIGHT_TYPE_flt", "STAND_mvt", "ADEP_mvt", "ADES_mvt",
        ]))
    df = pd.concat(frames, ignore_index=True)
    print(f"  Total rows: {len(df):,}")
    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    dt = df["MVT_TIME_UTC_mvt"].dt
    df["day_of_week"] = dt.dayofweek          # 0=Monday
    df["hour_of_day"] = dt.hour
    df["time_block"] = pd.cut(dt.hour, bins=[0, 6, 9, 12, 15, 18, 21, 24],
                               labels=["night", "AM_peak", "midday", "PM_early", "PM_peak", "evening", "late"],
                               right=False)
    # Stand prefix as a proxy for terminal area (first 1-2 chars)
    df["stand_prefix"] = df["STAND_mvt"].str.extract(r"^([A-Za-z]+|\d{1,2})")
    return df


def main():
    df = load_data()
    df = add_derived_features(df)

    target = "TAXITIME_SEC_mvt"
    df = df[df[target] > 0]  # drop zeros — likely data errors

    print(f"\nTarget: {target}")
    print(f"  Mean: {df[target].mean():.0f}s ({df[target].mean()/60:.1f} min)")
    print(f"  Median: {df[target].median():.0f}s ({df[target].median()/60:.1f} min)")
    print(f"  Std: {df[target].std():.0f}s")

    # Variables with exactly 2 groups → Mann-Whitney U
    two_group_vars = ["PHASE_mvt"]

    # Variables with >2 groups → Kruskal-Wallis
    multi_group_vars = [
        "RUNWAY_mvt",
        "WK_TBL_CAT_flt",
        "MARKET_SEGMENT_flt",
        "FLIGHT_TYPE_flt",
        "day_of_week",
        "hour_of_day",
        "time_block",
        "stand_prefix",
        "ADEP_mvt",
        "ADES_mvt",
    ]

    results = []

    print("\nRunning tests...")
    for col in two_group_vars:
        r = mwu_test(df, col, target)
        if r:
            results.append(r)
            print(f"  Mann-Whitney U: {col}")

    for col in multi_group_vars:
        r = kw_test(df, col, target)
        if r:
            results.append(r)
            print(f"  Kruskal-Wallis: {col}")

    results_df = pd.DataFrame(results)
    results_df["effect_label"] = results_df["effect_size"].apply(classify_effect)
    results_df = results_df.sort_values("effect_size", ascending=False)

    print("\n" + "=" * 80)
    print("RESULTS — ranked by effect size (ε²)")
    print("=" * 80)
    print(f"{'Variable':<20} {'Groups':>6} {'N':>9} {'p-value':>12} {'Effect ε²':>10} {'Magnitude':<12}")
    print("-" * 80)
    for _, row in results_df.iterrows():
        p_str = f"{row['p_value']:.2e}" if row["p_value"] < 0.001 else f"{row['p_value']:.4f}"
        sig = "***" if row["p_value"] < 0.001 else ("**" if row["p_value"] < 0.01 else ("*" if row["p_value"] < 0.05 else "ns"))
        print(f"{row['variable']:<20} {int(row['n_groups']):>6} {int(row['n_obs']):>9} {p_str:>12} {row['effect_size']:>10.4f}  {row['effect_label']:<10} {sig}")

    print("\nSignificance: *** p<0.001  ** p<0.01  * p<0.05  ns=not significant")
    print("Effect size:  negligible <0.01 | small 0.01–0.04 | medium 0.04–0.14 | large >0.14")

    # Per-group medians for significant variables
    print("\n" + "=" * 80)
    print("GROUP MEDIANS for top variables (sorted by median taxi time)")
    print("=" * 80)
    top_vars = results_df[results_df["effect_size"] >= 0.01]["variable"].tolist()
    for col in top_vars[:6]:
        gm = df.groupby(col)[target].agg(["median", "mean", "count"])
        gm.columns = ["median_s", "mean_s", "n"]
        gm["median_min"] = (gm["median_s"] / 60).round(1)
        gm = gm.sort_values("median_s", ascending=False)
        print(f"\n{col} (top 10 by median):")
        print(gm.head(10)[["median_min", "n"]].rename(columns={"median_min": "median (min)"}).to_string())


if __name__ == "__main__":
    main()

"""Ablation test: validation RMSE impact of removing each feature one at a time.

Loads data and precomputes congestion signals once, then iterates through each
feature flag (single-seed training per test for speed). Expect ~3 min per test.
"""

import sys
from pathlib import Path

from sklearn.metrics import root_mean_squared_error

sys.path.insert(0, str(Path(__file__).parent))
import model

ALL_ON = {k: True for k in model.FEATURES}


def apply_flags(flags: dict) -> None:
    model.FEATURES.update(flags)
    model.CATEGORICAL_FEATURES = [
        f for f in model._CATEGORICAL if model.FEATURES.get(f, False)
    ]


def validate(df, features, target, seed=42) -> float:
    X_train, y_train, X_val, y_val = model.time_based_split(df, features, target)
    m = model.train(X_train, y_train, seed=seed)
    return root_mean_squared_error(y_val, m.predict(X_val))


def main():
    print("Loading data (once)...")
    movements = model.load_movements()
    df = model.load_training_data(movements)
    pool = model.build_congestion_pool(movements)
    weather = model.load_weather_cache()
    target = df["TAXITIME_SEC_mvt"].astype(float)
    print(f"  {len(df):,} departure rows")

    print("Computing congestion signals (once)...")
    congestion = model.compute_congestion_signal(df, pool, model.CONGESTION_WINDOW_MINUTES)
    congestion_short = model.compute_congestion_signal(df, pool, model.CONGESTION_WINDOW_SHORT_MINUTES)
    congestion_acceleration = congestion_short - congestion
    day_deviation = model.compute_day_deviation_ratio(df, pool)
    print("  Done\n")

    features_to_test = list(ALL_ON.keys())
    print(f"Running baseline + {len(features_to_test)} ablation tests (single seed each)\n")

    # Baseline — all features on
    apply_flags(ALL_ON)
    feats = model.build_features(df, congestion, day_deviation, congestion_acceleration, weather)
    print("Baseline: testing all features ON...")
    baseline = validate(df, feats, target)
    print(f"Baseline RMSE: {baseline:.1f}s\n")

    results = []
    for i, feature in enumerate(features_to_test, 1):
        print(f"Test {i}/{len(features_to_test)}: testing {feature} flag...")
        apply_flags({**ALL_ON, feature: False})
        feats = model.build_features(df, congestion, day_deviation, congestion_acceleration, weather)
        rmse = validate(df, feats, target)
        delta = rmse - baseline
        print(f"Test {i} RMSE: {rmse:.1f}s  ({delta:+.1f}s vs baseline)\n")
        results.append((feature, rmse, delta))

    apply_flags(ALL_ON)  # restore all flags

    print("=" * 58)
    print("SUMMARY — sorted by impact")
    print(f"  {'feature':<25} {'RMSE':>7}  {'delta':>7}  verdict")
    print("-" * 58)
    for feature, rmse, delta in sorted(results, key=lambda x: x[2], reverse=True):
        if delta > 2:
            verdict = "HURTS  (removing this raises RMSE)"
        elif delta < -2:
            verdict = "noise  (removing this lowers RMSE)"
        else:
            verdict = "neutral"
        print(f"  {feature:<25} {rmse:>7.1f}s  {delta:>+7.1f}s  {verdict}")
    print("-" * 58)
    print(f"  {'baseline (all ON)':<25} {baseline:>7.1f}s")


if __name__ == "__main__":
    main()

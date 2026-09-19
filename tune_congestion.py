"""Sweep over CONGESTION_WINDOW_MINUTES values. Updates learnings.md with results."""

from pathlib import Path

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import root_mean_squared_error

from model import (
    CATEGORICAL_FEATURES,
    MIN_DAY_FLIGHTS,
    build_features,
    compute_congestion_signal,
    compute_day_deviation_ratio,
    load_training_data,
    time_based_split,
)

LEARNINGS_FILE = Path(__file__).with_name("learnings.md")
WINDOW_VALUES = [30, 45, 60, 90]
PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "num_leaves": 127,
    "learning_rate": 0.05,
    "min_data_in_leaf": 50,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "verbose": -1,
}


def train_and_evaluate(X_train, y_train, X_val, y_val) -> float:
    train_set = lgb.Dataset(
        X_train, label=y_train, categorical_feature=CATEGORICAL_FEATURES, free_raw_data=False
    )
    val_set = lgb.Dataset(
        X_val, label=y_val, categorical_feature=CATEGORICAL_FEATURES, free_raw_data=False
    )
    model = lgb.train(
        PARAMS,
        train_set,
        num_boost_round=500,
        valid_sets=[val_set],
        callbacks=[
            lgb.early_stopping(stopping_rounds=30, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    return root_mean_squared_error(y_val, model.predict(X_val))


def append_to_learnings(results: list[dict]) -> None:
    best = min(results, key=lambda r: r["rmse"])
    lines = [
        "\n---\n",
        "## Congestion Window Sweep\n",
        "num_leaves=127, min_data_in_leaf=50, lr=0.05, early stopping 30 rounds\n\n",
        "| window_minutes | coverage (%) | RMSE (s) | RMSE (min) |\n",
        "|---|---|---|---|\n",
    ]
    for r in results:
        marker = " ← best" if r == best else ""
        lines.append(
            f"| {r['window_minutes']} | {r['coverage']:.1f} "
            f"| {r['rmse']:.1f} | {r['rmse']/60:.2f} |{marker}\n"
        )
    lines.append(f"\n**Best window:** {best['window_minutes']} min → RMSE {best['rmse']:.1f}s\n")
    with open(LEARNINGS_FILE, "a") as f:
        f.writelines(lines)
    print(f"\nResults appended to {LEARNINGS_FILE}")


def main() -> None:
    print("Loading training data...")
    df = load_training_data()
    print(f"  {len(df):,} rows\n")

    target = df["TAXITIME_SEC_mvt"].astype(float)

    print(f"{'Window':>8}  {'Coverage':>10}  {'RMSE (s)':>10}  {'RMSE (min)':>10}")
    print("-" * 48)

    results = []
    for window in WINDOW_VALUES:
        congestion = compute_congestion_signal(df, df, window)
        coverage = 100 * congestion.notna().sum() / len(df)
        day_deviation = compute_day_deviation_ratio(df, df)
        features = build_features(df, congestion, day_deviation)
        X_train, y_train, X_val, y_val = time_based_split(df, features, target)
        rmse = train_and_evaluate(X_train, y_train, X_val, y_val)
        results.append({"window_minutes": window, "coverage": coverage, "rmse": rmse})
        print(f"{window:>7}m  {coverage:>9.1f}%  {rmse:>10.1f}  {rmse/60:>10.2f}")

    best = min(results, key=lambda r: r["rmse"])
    print(f"\nBest: {best['window_minutes']} min → {best['rmse']:.1f}s ({best['rmse']/60:.2f} min)")
    append_to_learnings(results)


if __name__ == "__main__":
    main()

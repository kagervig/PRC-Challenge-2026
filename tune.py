"""Grid search over num_leaves and min_data_in_leaf. Updates learnings.md with results."""

from pathlib import Path

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import root_mean_squared_error

from model import (
    CONGESTION_WINDOW_MINUTES,
    CATEGORICAL_FEATURES,
    TRAINING_FILES,
    MIN_DAY_FLIGHTS,
    build_features,
    compute_congestion_signal,
    compute_day_deviation_ratio,
    load_training_data,
    time_based_split,
)

LEARNINGS_FILE = Path(__file__).with_name("learnings.md")

NUM_LEAVES_VALUES = [63, 127, 255]
MIN_DATA_IN_LEAF_VALUES = [20, 50, 100]
LEARNING_RATE = 0.05
MAX_ROUNDS = 1000  # early stopping will cut this short


def train_with_early_stopping(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    num_leaves: int,
    min_data_in_leaf: int,
) -> tuple[float, int]:
    train_set = lgb.Dataset(
        X_train, label=y_train, categorical_feature=CATEGORICAL_FEATURES, free_raw_data=False
    )
    val_set = lgb.Dataset(
        X_val, label=y_val, categorical_feature=CATEGORICAL_FEATURES, free_raw_data=False
    )
    params = {
        "objective": "regression",
        "metric": "rmse",
        "num_leaves": num_leaves,
        "learning_rate": LEARNING_RATE,
        "min_data_in_leaf": min_data_in_leaf,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "verbose": -1,
    }
    model = lgb.train(
        params,
        train_set,
        num_boost_round=MAX_ROUNDS,
        valid_sets=[val_set],
        callbacks=[
            lgb.early_stopping(stopping_rounds=30, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    preds = model.predict(X_val)
    rmse = root_mean_squared_error(y_val, preds)
    return rmse, model.best_iteration


def append_to_learnings(results: list[dict]) -> None:
    best = min(results, key=lambda r: r["rmse"])
    lines = [
        "\n---\n",
        "## Hyperparameter Grid Search\n",
        f"Learning rate: {LEARNING_RATE} | Early stopping: 30 rounds | Max rounds: {MAX_ROUNDS}\n\n",
        f"| num_leaves | min_data_in_leaf | best_round | RMSE (s) | RMSE (min) |\n",
        f"|---|---|---|---|---|\n",
    ]
    for r in sorted(results, key=lambda r: r["rmse"]):
        marker = " ← best" if r == best else ""
        lines.append(
            f"| {r['num_leaves']} | {r['min_data_in_leaf']} | {r['best_round']} "
            f"| {r['rmse']:.1f} | {r['rmse']/60:.2f} |{marker}\n"
        )
    lines.append(
        f"\n**Best config:** num_leaves={best['num_leaves']}, "
        f"min_data_in_leaf={best['min_data_in_leaf']} → RMSE {best['rmse']:.1f}s\n"
    )
    with open(LEARNINGS_FILE, "a") as f:
        f.writelines(lines)
    print(f"\nResults appended to {LEARNINGS_FILE}")


def main() -> None:
    print("Loading training data...")
    df = load_training_data()
    print(f"  {len(df):,} rows")

    print(f"Computing congestion signal (window={CONGESTION_WINDOW_MINUTES} min)...")
    congestion = compute_congestion_signal(df, df, CONGESTION_WINDOW_MINUTES)
    day_deviation = compute_day_deviation_ratio(df, df)

    features = build_features(df, congestion, day_deviation)
    target = df["TAXITIME_SEC_mvt"].astype(float)

    X_train, y_train, X_val, y_val = time_based_split(df, features, target)
    print(f"  Train: {len(X_train):,} | Val: {len(X_val):,}\n")

    total = len(NUM_LEAVES_VALUES) * len(MIN_DATA_IN_LEAF_VALUES)
    print(f"{'Run':<5} {'num_leaves':<12} {'min_data_in_leaf':<18} {'best_round':<12} {'RMSE (s)':<10} {'RMSE (min)'}")
    print("-" * 70)

    results = []
    run = 1
    for num_leaves in NUM_LEAVES_VALUES:
        for min_data in MIN_DATA_IN_LEAF_VALUES:
            rmse, best_round = train_with_early_stopping(
                X_train, y_train, X_val, y_val, num_leaves, min_data
            )
            results.append({
                "num_leaves": num_leaves,
                "min_data_in_leaf": min_data,
                "best_round": best_round,
                "rmse": rmse,
            })
            print(f"{run}/{total:<4} {num_leaves:<12} {min_data:<18} {best_round:<12} {rmse:<10.1f} {rmse/60:.2f}")
            run += 1

    best = min(results, key=lambda r: r["rmse"])
    print(f"\nBest: num_leaves={best['num_leaves']}, min_data_in_leaf={best['min_data_in_leaf']} → {best['rmse']:.1f}s ({best['rmse']/60:.2f} min)")

    append_to_learnings(results)


if __name__ == "__main__":
    main()

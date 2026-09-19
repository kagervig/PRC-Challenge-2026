"""Train and evaluate per-airport models. Compares aggregate RMSE to global model baseline."""

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error

from model import (
    CONGESTION_WINDOW_MINUTES,
    build_features,
    compute_congestion_signal,
    load_training_data,
    time_based_split,
)

CATEGORICAL_FEATURES_NO_AIRPORT = [
    "runway",
    "stand",
    "airline",
    "weight_class",
    "market_segment",
    "month",
]

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


def train_airport_model(X_train: pd.DataFrame, y_train: pd.Series) -> lgb.Booster:
    dataset = lgb.Dataset(
        X_train,
        label=y_train,
        categorical_feature=CATEGORICAL_FEATURES_NO_AIRPORT,
        free_raw_data=False,
    )
    return lgb.train(
        PARAMS,
        dataset,
        num_boost_round=500,
        callbacks=[lgb.log_evaluation(period=0)],
    )


def main() -> None:
    print("Loading training data...")
    df = load_training_data()
    print(f"  {len(df):,} rows\n")

    print(f"Computing congestion signal (window={CONGESTION_WINDOW_MINUTES} min)...")
    congestion = compute_congestion_signal(df, df, CONGESTION_WINDOW_MINUTES)

    features = build_features(df, congestion)
    # Drop airport — it's constant within each per-airport model
    features = features.drop(columns=["airport"])

    target = df["TAXITIME_SEC_mvt"].astype(float)

    # Use the same global time cutoff for all airports so results are comparable
    cutoff = df["MVT_TIME_UTC_mvt"].quantile(0.83)

    airports = sorted(df["ADEP_mvt"].unique())
    print(f"Training {len(airports)} per-airport models...\n")
    print(f"{'Airport':<10} {'Train':>8} {'Val':>8} {'RMSE (s)':>10} {'RMSE (min)':>12}")
    print("-" * 52)

    results = []
    for airport in airports:
        mask = df["ADEP_mvt"] == airport
        airport_features = features[mask]
        airport_target = target[mask]
        airport_df = df[mask]

        train_mask = airport_df["MVT_TIME_UTC_mvt"] < cutoff
        X_train = airport_features[train_mask]
        y_train = airport_target[train_mask]
        X_val = airport_features[~train_mask]
        y_val = airport_target[~train_mask]

        if len(X_train) < 500 or len(X_val) < 100:
            print(f"{airport:<10} {'(skipped — too little data)'}")
            continue

        model = train_airport_model(X_train, y_train)
        preds = model.predict(X_val)
        rmse = root_mean_squared_error(y_val, preds)
        results.append({"airport": airport, "n_train": len(X_train), "n_val": len(X_val), "rmse": rmse, "sq_err_sum": ((preds - y_val.values) ** 2).sum()})
        print(f"{airport:<10} {len(X_train):>8,} {len(X_val):>8,} {rmse:>10.1f} {rmse/60:>12.2f}")

    total_n = sum(r["n_val"] for r in results)
    total_mse = sum(r["sq_err_sum"] for r in results) / total_n
    total_rmse = total_mse ** 0.5

    print("-" * 52)
    print(f"{'Overall':<10} {'':>8} {total_n:>8,} {total_rmse:>10.1f} {total_rmse/60:>12.2f}")
    print()
    print("Global model baseline: 384.2s")
    print(f"Per-airport aggregate: {total_rmse:.1f}s  (delta: {total_rmse - 384.2:+.1f}s)")


if __name__ == "__main__":
    main()

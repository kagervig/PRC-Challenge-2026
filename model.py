"""Train a LightGBM model to predict taxi-out time and write a submission file."""

import glob
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error

# -- Config -------------------------------------------------------------------

CONGESTION_WINDOW_MINUTES = 60  # easy to tune
MIN_DAY_FLIGHTS = 5  # completed flights required before day_deviation_ratio fires

DATA_DIR = Path(__file__).parent
PREDICTIONS_DIR = DATA_DIR / "predictions"
TRAINING_FILES = sorted(glob.glob(str(DATA_DIR / "training_*.parquet")))
RANKING_FILE = DATA_DIR / "ranking.parquet"
SUBMISSION_TEMPLATE = DATA_DIR / "submitting.parquet"
TEAM_NAME = "unique-umbrella"
SUBMISSION_VERSION = 9

CATEGORICAL_FEATURES = [
    "airport",
    "runway",
    "stand",
    "airline",
    "weight_class",
    "market_segment",
    "month",
]

# -- Data loading -------------------------------------------------------------

def load_training_data() -> pd.DataFrame:
    """Load and concatenate all monthly training parquet files, returning departures only."""
    frames = [pd.read_parquet(f) for f in TRAINING_FILES]
    df = pd.concat(frames, ignore_index=True)
    # ARR rows have taxi-in time; this model only predicts taxi-out
    df = df[df["PHASE_mvt"] == "DEP"].copy()
    # Drop date-rollover artifacts: taxi times > 6 hours are tracking errors
    # where MVT_TIME crossed midnight but block time used the wrong date
    return df[df["TAXITIME_SEC_mvt"] <= 21600].copy()


# -- Congestion signal --------------------------------------------------------

def compute_congestion_signal(
    departures: pd.DataFrame,
    completed: pd.DataFrame,
    window_minutes: int,
) -> pd.Series:
    """
    For each departure, compute the mean taxi time of completed flights
    at the same airport in the preceding window_minutes.

    Uses vectorised cumulative sums — no Python loops over rows.
    Returns NaN when no completed flights fall in the window
    (LightGBM handles NaN natively).
    """
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    window_s = window_minutes * 60.0
    result = pd.Series(np.nan, index=departures.index, dtype=float)

    completed_valid = completed[completed["TAXITIME_SEC_mvt"].notna()].copy()

    for airport, dep_group in departures.groupby("ADEP_mvt"):
        hist = (
            completed_valid[completed_valid["ADEP_mvt"] == airport]
            .sort_values("MVT_TIME_UTC_mvt")
        )
        if hist.empty:
            continue

        hist_s = (hist["MVT_TIME_UTC_mvt"] - epoch).dt.total_seconds().values
        hist_taxi = hist["TAXITIME_SEC_mvt"].values.astype(float)

        ref_times = dep_group["AOBT_3_flt"].fillna(dep_group["MVT_TIME_UTC_mvt"])
        ref_s = (ref_times - epoch).dt.total_seconds().values

        hi = np.searchsorted(hist_s, ref_s, side="left")
        lo = np.searchsorted(hist_s, ref_s - window_s, side="left")

        valid = ~np.isnan(hist_taxi)
        cum_sum = np.concatenate([[0.0], np.cumsum(np.where(valid, hist_taxi, 0.0))])
        cum_count = np.concatenate([[0], np.cumsum(valid.astype(int))])

        window_sum = cum_sum[hi] - cum_sum[lo]
        window_count = cum_count[hi] - cum_count[lo]
        safe_count = np.where(window_count > 0, window_count, 1)
        signals = np.where(window_count > 0, window_sum / safe_count, np.nan)

        result.loc[dep_group.index] = signals

    return result


def compute_day_deviation_ratio(
    departures: pd.DataFrame,
    completed: pd.DataFrame,
    min_flights: int = MIN_DAY_FLIGHTS,
) -> pd.Series:
    """
    For each departure, compute how today's taxi times at that airport compare
    to its long-run mean — a ratio of 1.5 means the airport is running 50%
    slower than normal today.

    Uses the same vectorised cumulative-sum pattern as compute_congestion_signal,
    but the window is from midnight to the departure time rather than a rolling
    lookback. Returns NaN until min_flights have completed that day (LightGBM
    handles NaN natively).
    """
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    result = pd.Series(np.nan, index=departures.index, dtype=float)

    completed_valid = completed[completed["TAXITIME_SEC_mvt"].notna()].copy()

    airport_baselines = completed_valid.groupby("ADEP_mvt")["TAXITIME_SEC_mvt"].mean()

    for airport, dep_group in departures.groupby("ADEP_mvt"):
        if airport not in airport_baselines.index:
            continue
        baseline = airport_baselines[airport]

        hist = (
            completed_valid[completed_valid["ADEP_mvt"] == airport]
            .sort_values("MVT_TIME_UTC_mvt")
        )
        if hist.empty:
            continue

        hist_s = (hist["MVT_TIME_UTC_mvt"] - epoch).dt.total_seconds().values
        hist_taxi = hist["TAXITIME_SEC_mvt"].values.astype(float)

        ref_times = dep_group["AOBT_3_flt"].fillna(dep_group["MVT_TIME_UTC_mvt"])
        ref_s = (ref_times - epoch).dt.total_seconds().values
        # Midnight UTC of each departure's date — defines the start of "today"
        day_start_s = (ref_times.dt.floor("D") - epoch).dt.total_seconds().values

        hi = np.searchsorted(hist_s, ref_s, side="left")
        lo = np.searchsorted(hist_s, day_start_s, side="left")

        valid = ~np.isnan(hist_taxi)
        cum_sum = np.concatenate([[0.0], np.cumsum(np.where(valid, hist_taxi, 0.0))])
        cum_count = np.concatenate([[0], np.cumsum(valid.astype(int))])

        window_sum = cum_sum[hi] - cum_sum[lo]
        window_count = cum_count[hi] - cum_count[lo]

        # Avoid divide-by-zero warning: np.where evaluates both branches before
        # selecting, so replace zero counts with 1 as a safe denominator (the
        # outer condition ensures those slots are always overwritten with NaN).
        safe_count = np.where(window_count > 0, window_count, 1)
        ratios = np.where(
            window_count >= min_flights,
            (window_sum / safe_count) / baseline,
            np.nan,
        )
        result.loc[dep_group.index] = ratios

    return result


# -- Feature engineering ------------------------------------------------------

def build_features(
    df: pd.DataFrame,
    congestion: pd.Series,
    day_deviation: pd.Series,
) -> pd.DataFrame:
    """Construct the feature matrix from raw movement and flight plan columns."""
    out = pd.DataFrame(index=df.index)
    out["airport"] = df["ADEP_mvt"].astype("category")
    out["runway"] = df["RUNWAY_mvt"].astype("category")
    out["stand"] = df["STAND_mvt"].astype("category")
    out["airline"] = df["AIRCRAFT_OPERATOR_flt"].astype("category")
    out["weight_class"] = df["WK_TBL_CAT_flt"].astype("category")
    out["market_segment"] = df["MARKET_SEGMENT_flt"].astype("category")
    out["month"] = df["MVT_TIME_UTC_mvt"].dt.month.astype("category")
    out["hour"] = df["MVT_TIME_UTC_mvt"].dt.hour
    out["gate_delay_sec"] = (
        (df["AOBT_3_flt"] - df["EOBT_1_flt"]).dt.total_seconds().fillna(0)
    )
    out["schedule_delay_sec"] = (
        (df["BLOCK_TIME_UTC_mvt"] - df["SCHED_TIME_UTC_mvt"]).dt.total_seconds()
    )
    out["congestion_signal"] = congestion
    out["day_deviation_ratio"] = day_deviation
    out["arvt_update_sec"] = (
        (df["ARVT_3_flt"] - df["ARVT_1_flt"]).dt.total_seconds()
    )
    return out


# -- Training -----------------------------------------------------------------

def train(features: pd.DataFrame, target: pd.Series, alpha: float | None = None) -> lgb.Booster:
    """Fit a LightGBM regression model and return the trained booster."""
    dataset = lgb.Dataset(
        features,
        label=target,
        categorical_feature=CATEGORICAL_FEATURES,
        # Retains the raw DataFrame so predict() can be called on the same features later
        free_raw_data=False,
    )
    params = {
        "objective": "regression" if alpha is None else "huber",
        "metric": "rmse",
        "num_leaves": 127,
        "learning_rate": 0.05,
        "min_data_in_leaf": 50,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "verbose": -1,
    }
    if alpha is not None:
        params["alpha"] = alpha
    model = lgb.train(
        params,
        dataset,
        num_boost_round=500,
        callbacks=[lgb.log_evaluation(period=50)],
    )
    return model


# -- Validation ---------------------------------------------------------------

def time_based_split(
    df: pd.DataFrame, features: pd.DataFrame, target: pd.Series
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Train on first 10 months, validate on last 2."""
    # 10/12 ≈ 0.833 — splits chronologically rather than randomly to avoid
    # leaking future patterns into the training set
    cutoff = df["MVT_TIME_UTC_mvt"].quantile(0.83)
    train_mask = df["MVT_TIME_UTC_mvt"] < cutoff
    return (
        features[train_mask],
        target[train_mask],
        features[~train_mask],
        target[~train_mask],
    )


# -- Submission ---------------------------------------------------------------

def write_submission(model: lgb.Booster, train_df: pd.DataFrame, version: int) -> Path:
    """Generate predictions for the ranking set and write the versioned submission parquet."""
    ranking = pd.read_parquet(RANKING_FILE)
    deps = ranking[ranking["PHASE_mvt"] == "DEP"].copy()

    # Ranking ARR rows have known taxi-in times at ADES_mvt — use them as same-airport
    # congestion context alongside training departures. Map ADES_mvt → ADEP_mvt so the
    # congestion function can match airports consistently.
    arrs = ranking[ranking["PHASE_mvt"] == "ARR"]
    arr_context = arrs[["MVT_TIME_UTC_mvt", "TAXITIME_SEC_mvt"]].copy()
    arr_context["ADEP_mvt"] = arrs["ADES_mvt"].values
    completed_pool = pd.concat([train_df, arr_context], ignore_index=True)

    congestion = compute_congestion_signal(deps, completed_pool, CONGESTION_WINDOW_MINUTES)
    day_deviation = compute_day_deviation_ratio(deps, completed_pool)
    features = build_features(deps, congestion, day_deviation)
    predictions = model.predict(features)

    template = pd.read_parquet(SUBMISSION_TEMPLATE)
    result = template.copy()

    # Map by MVT_ID rather than positional index because ranking and submission
    # template may not share the same row order
    pred_series = pd.Series(predictions, index=deps.index)
    mvt_to_pred = dict(zip(deps["MVT_ID_mvt"], pred_series.values))
    result["TAXITIME_SEC_mvt"] = result["MVT_ID_mvt"].map(mvt_to_pred)

    PREDICTIONS_DIR.mkdir(exist_ok=True)
    out_path = PREDICTIONS_DIR / f"{TEAM_NAME}_v{version}.parquet"
    result.to_parquet(out_path, index=False)
    print(f"Submission written to {out_path}")
    return out_path


# -- Main ---------------------------------------------------------------------

def main() -> None:
    """Run the full pipeline: load data, validate, train on all data, write submission."""
    print("Loading training data...")
    df = load_training_data()
    print(f"  {len(df):,} departure rows across {len(TRAINING_FILES)} files")

    print(f"Computing congestion signal (window={CONGESTION_WINDOW_MINUTES} min)...")
    congestion = compute_congestion_signal(df, df, CONGESTION_WINDOW_MINUTES)
    print(f"  Signal populated for {congestion.notna().sum():,} / {len(df):,} rows")

    print(f"Computing day deviation ratio (min_flights={MIN_DAY_FLIGHTS})...")
    day_deviation = compute_day_deviation_ratio(df, df)
    print(f"  Signal populated for {day_deviation.notna().sum():,} / {len(df):,} rows")

    features = build_features(df, congestion, day_deviation)
    target = df["TAXITIME_SEC_mvt"].astype(float)

    print("Splitting for validation...")
    X_train, y_train, X_val, y_val = time_based_split(df, features, target)
    print(f"  Train: {len(X_train):,} rows  |  Val: {len(X_val):,} rows")

    print("Training model on validation split...")
    val_model = train(X_train, y_train)
    val_preds = val_model.predict(X_val)
    rmse = root_mean_squared_error(y_val, val_preds)
    print(f"\nValidation RMSE: {rmse:.1f} seconds ({rmse/60:.2f} minutes)\n")

    print("Training final model on all data...")
    final_model = train(features, target)

    print("Writing submission file...")
    write_submission(final_model, df, SUBMISSION_VERSION)


if __name__ == "__main__":
    main()

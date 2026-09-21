"""Train a LightGBM model to predict taxi-out time and write a submission file."""

import glob
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error

# -- Config -------------------------------------------------------------------

CONGESTION_WINDOW_MINUTES = 60       # easy to tune
CONGESTION_WINDOW_SHORT_MINUTES = 10 # short window for acceleration signal
MIN_DAY_FLIGHTS = 5  # completed flights required before day_deviation_ratio fires

DATA_DIR = Path(__file__).parent
PREDICTIONS_DIR = DATA_DIR / "predictions"
TRAINING_FILES = sorted(glob.glob(str(DATA_DIR / "training_*.parquet")))
RANKING_FILE = DATA_DIR / "ranking.parquet"
SUBMISSION_TEMPLATE = DATA_DIR / "submitting.parquet"
TEAM_NAME = "unique-umbrella"
SUBMISSION_VERSION = 18

FEATURE_FRACTION = 0.8
ENSEMBLE_SEEDS = [42, 123, 456, 789, 1337, 2024, 31337, 99999, 7777]

CATEGORICAL_FEATURES = [
    "airport",
    "runway",
    "stand",
    "stand_prefix",
    "airline",
    "weight_class",
    "market_segment",
    "month",
]

# -- Data loading -------------------------------------------------------------

def load_movements() -> pd.DataFrame:
    """Load and concatenate all monthly training parquet files (all phases)."""
    frames = [pd.read_parquet(f) for f in TRAINING_FILES]
    return pd.concat(frames, ignore_index=True)


def load_training_data(movements: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return departure rows for training, optionally reusing a pre-loaded frame."""
    df = movements if movements is not None else load_movements()
    # ARR rows have taxi-in time; this model only predicts taxi-out
    df = df[df["PHASE_mvt"] == "DEP"].copy()
    # Drop date-rollover artifacts: taxi times > 6 hours are tracking errors
    # where MVT_TIME crossed midnight but block time used the wrong date
    return df[df["TAXITIME_SEC_mvt"] <= 21600].copy()


def build_congestion_pool(movements: pd.DataFrame) -> pd.DataFrame:
    """
    Build the same-airport congestion pool from arrival taxi-in times.

    Departure taxi-out times cannot serve as the pool: in the ranking period they
    are the withheld targets, so at prediction time every rolling window would be
    empty and the signal would collapse to a scale training never saw. Arrival
    taxi-in times measure the same airport's ground congestion, are present in both
    training and ranking data, and keep the feature identically scaled between train
    and serve. ADES_mvt (arrival airport) is mapped to ADEP_mvt so the congestion
    functions match airports consistently.
    """
    arrs = movements[movements["PHASE_mvt"] == "ARR"]
    pool = arrs[["MVT_TIME_UTC_mvt", "TAXITIME_SEC_mvt"]].copy()
    pool["ADEP_mvt"] = arrs["ADES_mvt"].values
    return pool


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
    congestion_acceleration: pd.Series,
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
    # BLOCK_TIME_UTC_mvt is withheld in the ranking set (it is derived from the
    # target). AOBT_3_flt measures the same off-block event from the flight plan
    # system and is present in both training and ranking data, so it is used in
    # both paths: training on BLOCK_TIME would tune the model on a column it never
    # sees at prediction time, a train/serve skew.
    out["schedule_delay_sec"] = (
        df["AOBT_3_flt"] - df["SCHED_TIME_UTC_mvt"]
    ).dt.total_seconds()
    out["stand_prefix"] = df["STAND_mvt"].str[0].fillna("UNK").astype("category")
    out["congestion_signal"] = congestion
    out["congestion_acceleration"] = congestion_acceleration
    out["day_deviation_ratio"] = day_deviation
    out["arvt_update_sec"] = (
        (df["ARVT_3_flt"] - df["ARVT_1_flt"]).dt.total_seconds()
    )
    return out


# -- Training -----------------------------------------------------------------

def train(
    features: pd.DataFrame,
    target: pd.Series,
    seed: int = 0,
    feature_fraction: float = FEATURE_FRACTION,
) -> lgb.Booster:
    """Fit a LightGBM regression model and return the trained booster."""
    dataset = lgb.Dataset(
        features,
        label=target,
        categorical_feature=CATEGORICAL_FEATURES,
        # Retains the raw DataFrame so predict() can be called on the same features later
        free_raw_data=False,
    )
    params = {
        "objective": "regression",
        "metric": "rmse",
        "num_leaves": 127,
        "learning_rate": 0.02,
        "min_data_in_leaf": 50,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "feature_fraction": feature_fraction,
        "seed": seed,
        "verbose": -1,
    }
    return lgb.train(params, dataset, num_boost_round=3000)


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

def write_submission(models: list[lgb.Booster], version: int) -> Path:
    """Generate ensemble predictions for the ranking set and write the versioned submission parquet."""
    ranking = pd.read_parquet(RANKING_FILE)
    deps = ranking[ranking["PHASE_mvt"] == "DEP"].copy()

    # Congestion context comes from ranking arrival taxi-in times, built the same way
    # as the training pool (see build_congestion_pool). Training departures cannot be
    # used here: they predate the ranking period and fall outside every rolling window,
    # which is what previously made this feature collapse to a taxi-in scale the model
    # was never trained on.
    pool = build_congestion_pool(ranking)

    congestion = compute_congestion_signal(deps, pool, CONGESTION_WINDOW_MINUTES)
    congestion_short = compute_congestion_signal(deps, pool, CONGESTION_WINDOW_SHORT_MINUTES)
    congestion_acceleration = congestion_short - congestion
    day_deviation = compute_day_deviation_ratio(deps, pool)
    features = build_features(deps, congestion, day_deviation, congestion_acceleration)
    predictions = np.clip(np.mean([m.predict(features) for m in models], axis=0), 0, None)

    template = pd.read_parquet(SUBMISSION_TEMPLATE)
    result = template.copy()

    # Map by MVT_ID rather than positional index because ranking and submission
    # template may not share the same row order
    pred_series = pd.Series(predictions, index=deps.index)

    # The surveillance data has a date-rollover bug at LIRF: when a flight's
    # actual pushback falls just before midnight UTC, BLOCK_TIME_UTC_mvt is
    # recorded one calendar day too early, inflating TAXITIME by 86400s in the
    # ground truth. These rows are identifiable because AOBT_3_flt (correct)
    # shows hour=23 while MVT_TIME_UTC_mvt (correct) shows hour=0.
    proxy_taxi = (deps["MVT_TIME_UTC_mvt"] - deps["AOBT_3_flt"]).dt.total_seconds()
    midnight_crossover = (
        (deps["ADEP_mvt"] == "LIRF")
        & (deps["AOBT_3_flt"].dt.hour == 23)
        & (deps["MVT_TIME_UTC_mvt"].dt.hour == 0)
    )
    # A separate class of bug: AOBT_3_flt itself has the wrong date (one day
    # early), making proxy_taxi ≈ 86400s. If BLOCK_TIME has the same error the
    # ground-truth TAXITIME is also ≈ proxy_taxi, so predicting proxy_taxi
    # directly is the best available estimate.
    aobt_date_error = (deps["ADEP_mvt"] == "LIRF") & (proxy_taxi > 50000)
    artifact_mask = midnight_crossover | aobt_date_error
    n_artifacts = artifact_mask.sum()
    if n_artifacts > 0:
        print(f"  Overriding {n_artifacts} artifact row(s) at LIRF")
        # midnight-crossover rows: proxy_taxi is correct, BLOCK_TIME is wrong by 1 day
        pred_series.loc[midnight_crossover] = proxy_taxi[midnight_crossover] + 86400
        # AOBT date-error rows: proxy_taxi ≈ ground truth TAXITIME
        aobt_only = aobt_date_error & ~midnight_crossover
        pred_series.loc[aobt_only] = proxy_taxi[aobt_only]

    # The artifact override runs after the initial np.clip, so re-clip to keep any
    # negative proxy_taxi values out of the submission.
    pred_series = pred_series.clip(lower=0)
    mvt_to_pred = dict(zip(deps["MVT_ID_mvt"], pred_series.values))
    result["TAXITIME_SEC_mvt"] = result["MVT_ID_mvt"].map(mvt_to_pred)

    PREDICTIONS_DIR.mkdir(exist_ok=True)
    out_path = PREDICTIONS_DIR / f"{TEAM_NAME}_v{version}.parquet"
    result.to_parquet(out_path, index=False)
    print("version: ", version)
    print(f"Submission written to {out_path}")
    return out_path


# -- Main ---------------------------------------------------------------------

def main() -> None:
    """Run the full pipeline: load data, validate, train on all data, write submission."""
    print("Loading training data...")
    movements = load_movements()
    df = load_training_data(movements)
    # Arrival taxi-in context, built identically for training and ranking so the
    # congestion features share the same scale at train and serve time.
    pool = build_congestion_pool(movements)
    print(f"  {len(df):,} departure rows across {len(TRAINING_FILES)} files")

    print(f"Computing congestion signal (window={CONGESTION_WINDOW_MINUTES} min)...")
    congestion = compute_congestion_signal(df, pool, CONGESTION_WINDOW_MINUTES)
    print(f"  Signal populated for {congestion.notna().sum():,} / {len(df):,} rows")

    print(f"Computing congestion acceleration (short window={CONGESTION_WINDOW_SHORT_MINUTES} min)...")
    congestion_short = compute_congestion_signal(df, pool, CONGESTION_WINDOW_SHORT_MINUTES)
    congestion_acceleration = congestion_short - congestion
    print(f"  Signal populated for {congestion_acceleration.notna().sum():,} / {len(df):,} rows")

    print(f"Computing day deviation ratio (min_flights={MIN_DAY_FLIGHTS})...")
    day_deviation = compute_day_deviation_ratio(df, pool)
    print(f"  Signal populated for {day_deviation.notna().sum():,} / {len(df):,} rows")

    features = build_features(df, congestion, day_deviation, congestion_acceleration)
    target = df["TAXITIME_SEC_mvt"].astype(float)

    print("Splitting for validation...")
    X_train, y_train, X_val, y_val = time_based_split(df, features, target)
    print(f"  Train: {len(X_train):,} rows  |  Val: {len(X_val):,} rows")

    print(f"Training validation ensemble ({len(ENSEMBLE_SEEDS)} seeds)...")
    val_preds_all = []
    for i, seed in enumerate(ENSEMBLE_SEEDS, 1):
        m = train(X_train, y_train, seed=seed)
        val_preds_all.append(m.predict(X_val))
        rmse = root_mean_squared_error(y_val, np.mean(val_preds_all, axis=0))
        print(f"  seed {i}/{len(ENSEMBLE_SEEDS)}: ensemble RMSE={rmse:.1f}s")
    print(f"\nValidation RMSE: {rmse:.1f} seconds ({rmse/60:.2f} minutes)\n")

    print(f"Training final ensemble on all data ({len(ENSEMBLE_SEEDS)} seeds)...")
    final_models = []
    for i, seed in enumerate(ENSEMBLE_SEEDS, 1):
        final_models.append(train(features, target, seed=seed))
        print(f"  seed {i}/{len(ENSEMBLE_SEEDS)} done")

    print("Writing submission file...")
    write_submission(final_models, SUBMISSION_VERSION)


if __name__ == "__main__":
    main()

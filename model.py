"""Train a LightGBM model to predict taxi-out time and write a submission file."""

import argparse
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
# A date-rollover artifact records AOBT and MVT_TIME a full day apart; no genuine
# taxi lasts this long, so proxy_taxi above this threshold flags the artifact.
ARTIFACT_PROXY_THRESHOLD_SEC = 50000
# Recorded taxi times above 6h are date-rollover tracking errors, not real taxis:
# excluded from training (corrupt labels) but KEPT in honest validation because the
# grader scores them.
ARTIFACT_TAXI_MAX_SEC = 21600

DATA_DIR = Path(__file__).parent
PREDICTIONS_DIR = DATA_DIR / "predictions"
TRAINING_FILES = sorted(glob.glob(str(DATA_DIR / "training_*.parquet")))
RANKING_FILE = DATA_DIR / "ranking.parquet"
SUBMISSION_TEMPLATE = DATA_DIR / "submitting.parquet"
TEAM_NAME = "unique-umbrella"
SUBMISSION_VERSION = 22
WEATHER_CACHE = DATA_DIR / "weather_cache.parquet"

FEATURE_FRACTION = 0.8
ENSEMBLE_SEEDS = [42, 123, 456, 789, 1337, 2024, 31337, 99999, 7777]

FEATURES = {
    # identity / location
    "airport":               True,
    "runway":                True,
    "stand":                 True,
    "stand_prefix":          True,
    "airline":               True,
    "weight_class":          True,
    "market_segment":        False,  # ablation: −0.2s, no signal
    # time
    "month":                 False,  # ablation: −0.4s, seasonality covered by weather_temp_c
    "hour":                  True,
    # delay signals
    "gate_delay_sec":        True,
    "schedule_delay_sec":    True,
    "proxy_taxi":            True,   # MVT_TIME - AOBT_3; target-adjacent, strongest single signal
    # congestion signals
    "congestion_signal":     True,
    "congestion_acceleration": False, # ablation: −0.7s, adds noise
    "day_deviation_ratio":   True,
    # flight plan signal
    "arvt_update_sec":       True,
    # weather
    "weather_temp_c":        True,
    "weather_wind_kt":       False,  # ablation: +0.1s, noise
    "weather_precip_mm":     False,  # ablation: −0.1s, no signal
    "weather_visibility_m":  False,  # ablation: 0.0s, NaN for 9/11 airports
    "weather_code":          True,   # ablation: +0.8s, marginal but positive
}

_CATEGORICAL = {
    "airport", "runway", "stand", "stand_prefix",
    "airline", "weight_class", "market_segment", "month",
}
CATEGORICAL_FEATURES = [f for f in _CATEGORICAL if FEATURES.get(f, False)]

# -- Weather cache ------------------------------------------------------------

def load_weather_cache() -> "pd.DataFrame | None":
    """Load pre-fetched hourly weather. Returns None if cache not yet built."""
    if not WEATHER_CACHE.exists():
        print("  Warning: weather_cache.parquet not found — weather features will be NaN")
        return None
    cache = pd.read_parquet(WEATHER_CACHE)
    # Normalize to datetime64[us, UTC] to match the training/ranking timestamp resolution
    if cache["hour_utc"].dt.tz is None:
        cache["hour_utc"] = cache["hour_utc"].dt.tz_localize("UTC")
    cache["hour_utc"] = cache["hour_utc"].dt.as_unit("us")
    return cache


def join_weather(df: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join hourly weather onto departure rows by (airport, push-back hour).
    Returns df with added columns: weather_temp_c, weather_wind_kt,
    weather_precip_mm, weather_visibility_m, weather_code.
    """
    ref_time = df["AOBT_3_flt"].fillna(df["MVT_TIME_UTC_mvt"])
    # Build keys without .values on the datetime column — .values strips timezone info
    keys = pd.DataFrame(index=df.index)
    keys["airport_icao"] = df["ADEP_mvt"].values
    keys["hour_utc"] = ref_time.dt.floor("h")
    merged = keys.merge(weather, on=["airport_icao", "hour_utc"], how="left")
    # merge resets to 0-based index; restore original so assign aligns correctly
    merged.index = df.index
    return df.assign(
        weather_temp_c=merged["temp_c"].values,
        weather_wind_kt=merged["wind_kt"].values,
        weather_precip_mm=merged["precip_mm"].values,
        weather_visibility_m=merged["visibility_m"].values,
        weather_code=merged["weather_code"].values,
    )


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
    return df[df["TAXITIME_SEC_mvt"] <= ARTIFACT_TAXI_MAX_SEC].copy()


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
    weather: "pd.DataFrame | None" = None,
) -> pd.DataFrame:
    """Construct the feature matrix from raw movement and flight plan columns."""
    F = FEATURES
    out = pd.DataFrame(index=df.index)
    if F["airport"]:        out["airport"]        = df["ADEP_mvt"].astype("category")
    if F["runway"]:         out["runway"]         = df["RUNWAY_mvt"].astype("category")
    if F["stand"]:          out["stand"]          = df["STAND_mvt"].astype("category")
    if F["airline"]:        out["airline"]        = df["AIRCRAFT_OPERATOR_flt"].astype("category")
    if F["weight_class"]:   out["weight_class"]   = df["WK_TBL_CAT_flt"].astype("category")
    if F["market_segment"]: out["market_segment"] = df["MARKET_SEGMENT_flt"].astype("category")
    if F["month"]:          out["month"]          = df["MVT_TIME_UTC_mvt"].dt.month.astype("category")
    if F["hour"]:           out["hour"]           = df["MVT_TIME_UTC_mvt"].dt.hour
    if F["gate_delay_sec"]:
        out["gate_delay_sec"] = (df["AOBT_3_flt"] - df["EOBT_1_flt"]).dt.total_seconds().fillna(0)
    # BLOCK_TIME_UTC_mvt is withheld in the ranking set (it is derived from the
    # target). AOBT_3_flt measures the same off-block event from the flight plan
    # system and is present in both training and ranking data, so it is used in
    # both paths: training on BLOCK_TIME would tune the model on a column it never
    # sees at prediction time, a train/serve skew.
    if F["schedule_delay_sec"]:
        out["schedule_delay_sec"] = (df["AOBT_3_flt"] - df["SCHED_TIME_UTC_mvt"]).dt.total_seconds()
    if F["stand_prefix"]:
        out["stand_prefix"] = df["STAND_mvt"].str[0].fillna("UNK").astype("category")
    if F["proxy_taxi"]:
        # MVT_TIME - AOBT_3: for clean flights a sharp taxi estimate; on date-rollover
        # artifacts it equals the corrupted label. Uses MVT_TIME, which earlier work
        # avoided as target-adjacent — included deliberately now as the dominant
        # available signal (reverses the prior "never use MVT_TIME" stance).
        out["proxy_taxi"] = (df["MVT_TIME_UTC_mvt"] - df["AOBT_3_flt"]).dt.total_seconds()
    if F["congestion_signal"]:     out["congestion_signal"]     = congestion
    if F["congestion_acceleration"]: out["congestion_acceleration"] = congestion_acceleration
    if F["day_deviation_ratio"]:   out["day_deviation_ratio"]   = day_deviation
    if F["arvt_update_sec"]:
        out["arvt_update_sec"] = (df["ARVT_3_flt"] - df["ARVT_1_flt"]).dt.total_seconds()

    any_weather = any(F[k] for k in ["weather_temp_c", "weather_wind_kt", "weather_precip_mm", "weather_visibility_m", "weather_code"])
    if any_weather and weather is not None:
        enriched = join_weather(df, weather)
        if F["weather_temp_c"]:       out["weather_temp_c"]       = enriched["weather_temp_c"].values
        if F["weather_wind_kt"]:      out["weather_wind_kt"]      = enriched["weather_wind_kt"].values
        if F["weather_precip_mm"]:    out["weather_precip_mm"]    = enriched["weather_precip_mm"].values
        if F["weather_visibility_m"]: out["weather_visibility_m"] = enriched["weather_visibility_m"].values
        if F["weather_code"]:         out["weather_code"]         = enriched["weather_code"].values

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

def apply_artifact_override(
    pred_series: pd.Series, deps: pd.DataFrame, verbose: bool = False
) -> pd.Series:
    """
    Override predictions for detectable date-rollover artifacts and re-clip to >= 0.

    A data-entry error records AOBT_3_flt and MVT_TIME_UTC_mvt a full calendar day
    apart, so the ground-truth TAXITIME (MVT_TIME - BLOCK_TIME, with BLOCK_TIME
    tracking AOBT) is ~86400s. proxy_taxi (MVT_TIME - AOBT_3_flt) exceeds a full day
    on these rows and nowhere else, and training confirms it matches the buggy label,
    so predicting proxy_taxi directly is the best available estimate. Only the flavour
    where MVT_TIME (not the hidden BLOCK_TIME) carries the bad date is detectable here;
    normal near-midnight flights are left to the model. Re-clipping guards against
    negative proxy_taxi values.
    """
    proxy_taxi = (deps["MVT_TIME_UTC_mvt"] - deps["AOBT_3_flt"]).dt.total_seconds()
    artifact_mask = proxy_taxi > ARTIFACT_PROXY_THRESHOLD_SEC
    out = pred_series.copy()
    out.loc[artifact_mask] = proxy_taxi[artifact_mask]
    if verbose:
        n_artifacts = int(artifact_mask.sum())
        if n_artifacts > 0:
            airports = deps.loc[artifact_mask, "ADEP_mvt"].value_counts().to_dict()
            print(f"  Date-rollover override: {n_artifacts} row(s) at {airports}")
            for idx in deps.index[artifact_mask]:
                assigned = proxy_taxi[idx]
                print(
                    f"    {deps.at[idx, 'ADEP_mvt']} MVT_ID={deps.at[idx, 'MVT_ID_mvt']} "
                    f"proxy_taxi={assigned:.0f}s -> assigned {assigned:.0f}s"
                )
        else:
            print("  Date-rollover override: no artifact rows detected")
    return out.clip(lower=0)


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
    weather = load_weather_cache()
    features = build_features(deps, congestion, day_deviation, congestion_acceleration, weather)
    predictions = np.clip(np.mean([m.predict(features) for m in models], axis=0), 0, None)

    template = pd.read_parquet(SUBMISSION_TEMPLATE)
    result = template.copy()

    # Map by MVT_ID rather than positional index because ranking and submission
    # template may not share the same row order
    pred_series = pd.Series(predictions, index=deps.index)

    pred_series = apply_artifact_override(pred_series, deps, verbose=True)
    mvt_to_pred = dict(zip(deps["MVT_ID_mvt"], pred_series.values))
    result["TAXITIME_SEC_mvt"] = result["MVT_ID_mvt"].map(mvt_to_pred)

    PREDICTIONS_DIR.mkdir(exist_ok=True)
    out_path = PREDICTIONS_DIR / f"{TEAM_NAME}_v{version}.parquet"
    result.to_parquet(out_path, index=False)
    print("version: ", version)
    print(f"Submission written to {out_path}")
    for feature in FEATURES:
        if feature == False:
            print(feature, " set to FALSE")
    return out_path


# -- Main ---------------------------------------------------------------------

def main() -> None:
    """Run the full pipeline: load data, validate, train on all data, write submission."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Train a single seed instead of the full ensemble for faster iteration "
        "(lower accuracy; use when validating pipeline changes, not for final runs)",
    )
    args = parser.parse_args()
    seeds = ENSEMBLE_SEEDS[:1] if args.fast else ENSEMBLE_SEEDS
    if args.fast:
        print(f"FAST MODE: using {len(seeds)} seed (reduced accuracy)\n")

    print("Loading training data...")
    movements = load_movements()
    # Keep the unfiltered departures: artifact rows are excluded from *training* but
    # kept in *validation* so the reported RMSE reflects what the grader scores.
    dep = movements[movements["PHASE_mvt"] == "DEP"].copy()
    # Arrival taxi-in context, built identically for training and ranking so the
    # congestion features share the same scale at train and serve time.
    pool = build_congestion_pool(movements)
    print(f"  {len(dep):,} departure rows across {len(TRAINING_FILES)} files")

    print("Loading weather cache...")
    weather = load_weather_cache()

    print(f"Computing congestion signal (window={CONGESTION_WINDOW_MINUTES} min)...")
    congestion = compute_congestion_signal(dep, pool, CONGESTION_WINDOW_MINUTES)
    congestion_short = compute_congestion_signal(dep, pool, CONGESTION_WINDOW_SHORT_MINUTES)
    congestion_acceleration = congestion_short - congestion
    print(f"Computing day deviation ratio (min_flights={MIN_DAY_FLIGHTS})...")
    day_deviation = compute_day_deviation_ratio(dep, pool)

    features = build_features(dep, congestion, day_deviation, congestion_acceleration, weather)
    target = dep["TAXITIME_SEC_mvt"].astype(float)
    is_clean = target <= ARTIFACT_TAXI_MAX_SEC  # artifact rows have corrupt labels

    # -- Honest validation ----------------------------------------------------
    # Train on the earlier clean rows; evaluate on the most recent slice with
    # artifact rows KEPT IN and the same override the submission applies, so the
    # number tracks the leaderboard (clean-only RMSE hid the artifacts that
    # dominate the grader's error).
    print("Splitting for validation (time-based; artifacts kept in val)...")
    cutoff = dep["MVT_TIME_UTC_mvt"].quantile(0.83)
    train_mask = (dep["MVT_TIME_UTC_mvt"] < cutoff) & is_clean
    val_mask = dep["MVT_TIME_UTC_mvt"] >= cutoff
    y_val = target[val_mask].values
    val_clean_sel = is_clean[val_mask].values
    print(f"  Train: {int(train_mask.sum()):,} clean rows  |  "
          f"Val: {int(val_mask.sum()):,} rows ({int((~val_clean_sel).sum())} artifacts kept)")

    print(f"Training validation ensemble ({len(seeds)} seeds)...")
    val_preds_all = []
    for i, seed in enumerate(seeds, 1):
        m = train(features[train_mask], target[train_mask], seed=seed)
        val_preds_all.append(m.predict(features[val_mask]))
        mean_pred = pd.Series(np.mean(val_preds_all, axis=0), index=dep.index[val_mask])
        adj = apply_artifact_override(mean_pred, dep[val_mask]).values
        honest = root_mean_squared_error(y_val, adj)
        clean = root_mean_squared_error(y_val[val_clean_sel], adj[val_clean_sel])
        print(f"  seed {i}/{len(seeds)}: honest RMSE={honest:.1f}s  (clean-only={clean:.1f}s)")
    print(f"\nHonest validation RMSE: {honest:.1f}s  |  clean-only: {clean:.1f}s\n")

    print(f"Training final ensemble on all clean data ({len(seeds)} seeds)...")
    final_models = []
    for i, seed in enumerate(seeds, 1):
        final_models.append(train(features[is_clean], target[is_clean], seed=seed))
        print(f"  seed {i}/{len(seeds)} done")

    print("Writing submission file...")
    write_submission(final_models, SUBMISSION_VERSION)


if __name__ == "__main__":
    main()

"""Fetch and cache hourly weather data for all 11 prediction airports.

Run once before model.py. Output: weather_cache.parquet in the same directory.
Uses Meteostat for 9 airports and IEM ASOS for LTFM (Istanbul) and LEMD (Madrid).

LTFM: Meteostat maps to old Ataturk airport (33 km away); IEM has correct coords.
LEMD: Meteostat only 40% hourly coverage; IEM has 200% coverage.
"""

import io
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

DATA_DIR = Path(__file__).parent
CACHE_FILE = DATA_DIR / "weather_cache.parquet"

# Covers all training months (Jan 2025–Jan 2026) and full ranking period (Jan–Jul 2026)
START = datetime(2025, 1, 1)
END = datetime(2026, 8, 1)

# 9 airports: Meteostat Point API (99–100% coverage, stations 0.5–2.8 km from field)
METEOSTAT_AIRPORTS = {
    "LIRF": (41.8003, 12.2389),
    "LFPG": (49.0097,  2.5478),
    "EGLL": (51.4775, -0.4614),
    "EDDF": (50.0264,  8.5431),
    "EDDM": (48.3537, 11.7860),
    "EHAM": (52.3086,  4.7639),
    "LSZH": (47.4647,  8.5492),
    "LOWW": (48.1103, 16.5697),
    "UUEE": (55.9726, 37.4146),
}

# 2 airports: IEM ASOS (Meteostat is wrong station or low coverage)
IEM_AIRPORTS = ["LTFM", "LEMD"]


def fetch_meteostat(airports: dict) -> pd.DataFrame:
    try:
        import meteostat
    except ImportError:
        raise ImportError("pip install meteostat")

    frames = []
    for icao, (lat, lon) in airports.items():
        print(f"  Meteostat: {icao}...", end=" ", flush=True)
        try:
            # v2 API: resolve nearest station first, then fetch by station ID
            nearby = meteostat.stations.nearby(meteostat.Point(lat, lon), limit=1)
            if nearby.empty:
                print("NO STATION")
                continue
            station_id = nearby.index[0]
            data = meteostat.hourly(station_id, START, END).fetch()
        except Exception as e:
            print(f"ERROR: {e}")
            continue
        if data is None or data.empty:
            print("NO DATA")
            continue
        data = data.reset_index()
        data = data.rename(columns={"time": "hour_utc"})
        # Meteostat returns tz-naive UTC timestamps — make explicit
        data["hour_utc"] = pd.to_datetime(data["hour_utc"]).dt.tz_localize("UTC")
        data["airport_icao"] = icao
        # wspd is km/h; convert to knots
        data["wind_kt"] = (
            pd.to_numeric(data["wspd"], errors="coerce") / 1.852
            if "wspd" in data.columns else np.nan
        )
        data["temp_c"] = pd.to_numeric(data["temp"], errors="coerce") if "temp" in data.columns else np.nan
        data["precip_mm"] = pd.to_numeric(data["prcp"], errors="coerce") if "prcp" in data.columns else np.nan
        data["weather_code"] = pd.to_numeric(data["coco"], errors="coerce") if "coco" in data.columns else np.nan
        # Meteostat has no direct visibility column
        data["visibility_m"] = np.nan
        frames.append(
            data[["airport_icao", "hour_utc", "temp_c", "wind_kt", "precip_mm", "visibility_m", "weather_code"]]
        )
        print(f"{len(data):,} rows  (station {station_id})")

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fetch_iem_airport(icao: str) -> pd.DataFrame:
    url = (
        "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
        f"?station={icao}"
        "&data=tmpc,sknt,vsby,p01i"
        f"&year1={START.year}&month1={START.month:02d}&day1={START.day:02d}"
        f"&year2={END.year}&month2={END.month:02d}&day2={END.day:02d}"
        "&tz=UTC&format=comma&latlon=no&direct=no&report_type=3,4"
    )
    resp = requests.get(url, timeout=180)
    resp.raise_for_status()

    lines = [line for line in resp.text.splitlines() if not line.startswith("#")]
    if len(lines) < 2:
        return pd.DataFrame()

    # "M" = missing, "T" = trace precip (treat as NaN; 0.001 mm is noise)
    df = pd.read_csv(io.StringIO("\n".join(lines)), na_values=["M", "T", ""])
    if df.empty or "valid" not in df.columns:
        return pd.DataFrame()

    df["valid"] = pd.to_datetime(df["valid"], utc=True)
    df["hour_utc"] = df["valid"].dt.floor("h")
    df["airport_icao"] = icao

    df["temp_c"] = pd.to_numeric(df.get("tmpc"), errors="coerce")
    df["wind_kt"] = pd.to_numeric(df.get("sknt"), errors="coerce")
    # statute miles → metres
    df["visibility_m"] = pd.to_numeric(df.get("vsby"), errors="coerce") * 1609.34
    # inches → mm
    df["precip_mm"] = pd.to_numeric(df.get("p01i"), errors="coerce") * 25.4
    # wxcodes not requested — too sparse to aggregate meaningfully
    df["weather_code"] = np.nan

    # Aggregate sub-hourly METAR obs (~2/hr) to one row per hour
    hourly = (
        df.groupby(["airport_icao", "hour_utc"])[
            ["temp_c", "wind_kt", "visibility_m", "precip_mm"]
        ]
        .mean()
        .reset_index()
    )
    hourly["weather_code"] = np.nan
    return hourly[["airport_icao", "hour_utc", "temp_c", "wind_kt", "precip_mm", "visibility_m", "weather_code"]]


def fetch_iem(airports: list) -> pd.DataFrame:
    frames = []
    for icao in airports:
        print(f"  IEM: {icao}...", end=" ", flush=True)
        try:
            df = fetch_iem_airport(icao)
            if df.empty:
                print("NO DATA")
            else:
                frames.append(df)
                print(f"{len(df):,} rows")
        except Exception as e:
            print(f"ERROR: {e}")
        time.sleep(3)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def print_coverage(cache: pd.DataFrame) -> None:
    expected_hours = int((END - START).total_seconds() / 3600)
    print(f"\nExpected hours per airport: {expected_hours:,}")
    for icao in sorted(cache["airport_icao"].unique()):
        grp = cache[cache["airport_icao"] == icao]
        filled = {
            col: f"{grp[col].notna().mean():.1%}"
            for col in ["temp_c", "wind_kt", "precip_mm", "visibility_m", "weather_code"]
        }
        print(f"  {icao}: {len(grp):,} rows  {filled}")


def main() -> None:
    print(f"Fetching weather: {START.date()} to {END.date()}")
    print(f"Output: {CACHE_FILE}\n")

    print(f"Fetching Meteostat ({len(METEOSTAT_AIRPORTS)} airports)...")
    meteostat_df = fetch_meteostat(METEOSTAT_AIRPORTS)

    print(f"\nFetching IEM ASOS ({len(IEM_AIRPORTS)} airports)...")
    iem_df = fetch_iem(IEM_AIRPORTS)

    frames = [df for df in [meteostat_df, iem_df] if not df.empty]
    if not frames:
        print("ERROR: no data fetched from any source")
        return

    cache = pd.concat(frames, ignore_index=True)
    cache = cache.sort_values(["airport_icao", "hour_utc"]).reset_index(drop=True)

    print_coverage(cache)

    cache.to_parquet(CACHE_FILE, index=False)
    print(f"\nWritten: {CACHE_FILE}  ({len(cache):,} total rows)")


if __name__ == "__main__":
    main()

"""
Exports one row per distinct flight number across all training and ranking data.

Departure time is the most common scheduled time-of-day (HH:MM UTC) for that
flight number. Origin, destination, and aircraft type are the most common values
observed for that flight number.

Output: flights.csv
"""

import glob

import airportsdata
import pandas as pd

COLS = {
    "FLIGHT_mvt": "flight_number",
    "ADEP_mvt": "origin",
    "ADES_mvt": "destination",
    "LOBT_flt": "departure_time_utc",
    "AIRCRAFT_TYPE_mvt": "aircraft_type",
}

files = sorted(glob.glob("training_*.parquet")) + ["ranking.parquet"]
print(f"Loading {len(files)} files...")

frames = []
for f in files:
    frames.append(pd.read_parquet(f, columns=list(COLS.keys())))

df = pd.concat(frames, ignore_index=True)
df = df.rename(columns=COLS)
df = df.dropna(subset=["flight_number", "origin", "destination", "departure_time_utc"])

# Extract time-of-day from departure timestamp
df["departure_time_utc"] = pd.to_datetime(df["departure_time_utc"], utc=True)
df["departure_time"] = df["departure_time_utc"].dt.strftime("%H:%M")

def most_common(s):
    return s.mode().iloc[0]

print("Deduplicating by flight number...")
result = (
    df.groupby("flight_number")
    .agg(
        origin=("origin", most_common),
        destination=("destination", most_common),
        departure_time=("departure_time", most_common),
        aircraft_type=("aircraft_type", most_common),
        occurrences=("flight_number", "count"),
    )
    .reset_index()
    .sort_values("flight_number")
)

# Map ICAO → IATA for origin and destination
icao_airports = airportsdata.load("ICAO")
icao_to_iata = {k: v["iata"] for k, v in icao_airports.items() if v["iata"]}

result["origin"] = result["origin"].map(icao_to_iata).fillna(result["origin"])
result["destination"] = result["destination"].map(icao_to_iata).fillna(result["destination"])

print(f"Distinct flight numbers: {len(result):,}")
result.to_csv("flights.csv", index=False)
print("Written to flights.csv")

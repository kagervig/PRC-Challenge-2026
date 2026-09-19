import argparse
import os
import pydoc
from pathlib import Path

import pandas as pd
import psycopg2
from dotenv import load_dotenv

load_dotenv()


def get_db_connection() -> psycopg2.extensions.connection:
    url = os.environ["DATABASE_URL"]
    return psycopg2.connect(url)


# Change this filename if the input CSV has a different name or location.
DEFAULT_CSV = Path(__file__).with_name("flights.csv")

def load_flights(path: str | Path = DEFAULT_CSV) -> pd.DataFrame:
	"""Read flights.csv and return its contents as a DataFrame."""
	# Keep flight numbers as text so values such as "0111X" are not changed.
	flights = pd.read_csv(path, dtype={"flight_number": "string"})
	# Convert this column to numbers so comparisons and calculations work.
	flights["occurrences"] = pd.to_numeric(flights["occurrences"], errors="raise")

	# Add cleaning or calculated columns here. For example:
	# flights["route"] = flights["origin"] + "-" + flights["destination"]
	return flights

def filter_flights(
	flights: pd.DataFrame,
	*,
	origin: str | None = None,
	destination: str | None = None,
	aircraft_type: str | None = None,
	min_occurrences: int | None = None,
) -> pd.DataFrame:
	"""Return rows matching the supplied filters."""
	result = flights
	# Add another optional filter here when you need to filter a new column.
	if origin is not None:
		result = result[result["origin"].eq(origin.upper())]
	if destination is not None:
		result = result[result["destination"].eq(destination.upper())]
	if aircraft_type is not None:
		result = result[result["aircraft_type"].eq(aircraft_type.upper())]
	if min_occurrences is not None:
		result = result[result["occurrences"].ge(min_occurrences)]
	return result

def summarise_flights(flights: pd.DataFrame) -> pd.DataFrame:
	"""Summarise flight counts and occurrences by route."""
	# Change the groupby columns to summarise by a different combination.
	# Add more named calculations inside agg() when you need more totals.
	summary = (
		flights.groupby(["origin", "destination"], as_index=False)
		.agg(flight_count=("flight_number", "nunique"), occurrences=("occurrences", "sum"))
		.sort_values("occurrences", ascending=False)
	)
	summary["daily_average"] = summary["occurrences"] / 365
	return summary

def summarise_route(flights: pd.DataFrame) -> pd.DataFrame:
	origin = input("Origin: ").upper()
	destination = input("Destination: ").upper()
	route = flights[
		flights["origin"].eq(origin) & flights["destination"].eq(destination)
	]
	summary = (
		route.groupby("flight_number", as_index=False)
		.agg(
			occurrences=("occurrences", "sum"),
			aircraft_types=("aircraft_type", lambda x: ", ".join(sorted(x.dropna().unique()))),
		)
		.sort_values("occurrences", ascending=False)
	)
	summary["daily_average"] = summary["occurrences"] / 365
	return summary

def get_airports() -> list[str]:
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT iata_code FROM airports")
            rows = cur.fetchall()
    return [row[0] for row in rows]

def count_dodgy_flight_numbers(flights: pd.DataFrame):
    numbers = flights["flight_number"].dropna()
    endswith_char = set(numbers[numbers.apply(lambda fn: fn[-1].isalpha())])
    long_prefix = set(numbers[numbers.apply(lambda fn: len(fn) >= 3 and fn[:3].isalpha())])
    only_endswith = endswith_char - long_prefix
    only_long_prefix = long_prefix - endswith_char
    print(f"Ends with letter (total): {len(endswith_char):,} — exclusive: {len(only_endswith):,} — {sorted(only_endswith)[:10]}")
    print(f"Long prefix (total): {len(long_prefix):,} — exclusive: {len(only_long_prefix):,} — {sorted(only_long_prefix)[:10]}")
    print(f"In both: {len(endswith_char & long_prefix):,}")
    return endswith_char
    


def remove_dodgy_flight_numbers(flights: pd.DataFrame, path: str | Path) -> pd.DataFrame:
    numbers = flights["flight_number"].dropna()
    endswith_char = set(numbers[numbers.apply(lambda fn: fn[-1].isalpha())])
    long_prefix = set(numbers[numbers.apply(lambda fn: len(fn) >= 3 and fn[:3].isalpha())])
    dodgy = endswith_char | long_prefix
    cleaned = flights[~flights["flight_number"].isin(dodgy)].copy()
    cleaned.to_csv(path, index=False)
    print(f"Removed {len(flights) - len(cleaned):,} dodgy flight numbers. {len(cleaned):,} rows written to {path}.")
    return cleaned


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	# Add command-line options here when you want to control new operations.
	parser.add_argument("--file", type=Path, default=DEFAULT_CSV, help="CSV file to load")
	parser.add_argument("--origin", help="Filter by origin airport")
	parser.add_argument("--destination", help="Filter by destination airport")
	parser.add_argument("--aircraft-type", help="Filter by aircraft type")
	parser.add_argument("--min-occurrences", type=int, help="Keep flights with at least this many occurrences")
	parser.add_argument("--search", action="store_true", help="Search flights by origin and destination")
	parser.add_argument("--summary", action="store_true", help="Show totals grouped by route")
	parser.add_argument("--limit", type=int, help="Show only the first N matching rows")
	parser.add_argument("--dodgy", action="store_true", help="Lists flight numbers ending in alphabetical chars")
	parser.add_argument("--remove-dodgy", action="store_true", help="Remove dodgy flight numbers and overwrite the input file")
	return parser.parse_args()

def main() -> None:
    # This is the main workflow: load, filter, then display the result.
    args = parse_args()
    flights = load_flights(args.file)
    flights = filter_flights(
        flights,
        origin=args.origin,
        destination=args.destination,
        aircraft_type=args.aircraft_type,
        min_occurrences=args.min_occurrences,
    )
    if args.remove_dodgy:
        flights = remove_dodgy_flight_numbers(flights, args.file)
    elif args.dodgy:
        endswith_char = count_dodgy_flight_numbers(flights)
        #pydoc.pager("\n".join(sorted(endswith_char)))
    elif args.search:
        pydoc.pager(summarise_route(flights).to_string(index=False))
    else:
        pydoc.pager(summarise_flights(flights).to_string(index=False))

    
if __name__ == "__main__":
	main()
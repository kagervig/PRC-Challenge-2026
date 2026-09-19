"""Load and inspect the exported flight reference data."""

import argparse
import os
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
invalid_aircraft: set = {"A124","A139","A400","AS32","ASTR","C130","C17","C27J","C295","C30J","CL30","CL35","CL60","COL4","D328","DA42","DA62","E50P","E545","E550","E55P","EA50","F100","F2TH","F900","FA20","FA50","FA6X","FA7X","FA8X","G150","G280","GA5C","GA6C","GA7C","GALX","GL5T","GL7T","GLEX","GLF4","GLF5","GLF6","H25B","H25C","HA4T","HDJT","LJ31","LJ35","LJ40","LJ45","LJ55","LJ60","LJ70","LJ75","M600","P180","PC12","PC24","PRM1","S22T","SF50","SW3","SW4","TBM7","TBM8","TBM9","ZZZZ"}
ICAO_TO_IATA = {
    'A19N': '320', 'A20N': '320', 'A21N': '321', 'A306': 'AB3', 'A30B': 'AB3',
    'A310': '310', 'A318': '318', 'A319': '319', 'A320': '320', 'A321': '321',
    'A332': '332', 'A333': '333', 'A337': '330', 'A339': '330', 'A342': '342',
    'A343': '343', 'A345': '345', 'A346': '346', 'A359': '350', 'A35K': '350',
    'A388': '388', 'AN12': 'AN4', 'AN26': 'AN4', 'AT43': 'AT5', 'AT45': 'AT5',
    'AT72': 'AT7', 'AT73': 'AT7', 'AT75': 'AT7', 'AT76': 'AT7', 'B190': 'BE1',
    'B350': 'BEH', 'B38M': '73M', 'B39M': '73M', 'B733': '733', 'B734': '734',
    'B735': '735', 'B737': '737', 'B738': '738', 'B739': '739', 'B742': '747',
    'B744': '744', 'B748': '747', 'B752': '752', 'B753': '753', 'B762': '767',
    'B763': '763', 'B764': '764', 'B772': '772', 'B773': '773', 'B77L': '77L',
    'B77W': '77W', 'B788': '788', 'B789': '787', 'B78X': '787', 'BCS1': '220',
    'BCS3': '220', 'BE20': 'BEH', 'BE40': 'BEC', 'BE4W': 'BEC', 'BE58': 'BEC',
    'BE9L': 'BEH', 'C206': 'CNA', 'C208': 'CNA', 'C25A': 'CNA', 'C25B': 'CNA',
    'C25C': 'CNA', 'C25M': 'CNA', 'C408': 'CNA', 'C421': 'CNA', 'C425': 'CNA',
    'C510': 'CNA', 'C525': 'CNA', 'C550': 'CNA', 'C55B': 'CNA', 'C560': 'CNA',
    'C56X': 'CNA', 'C650': 'CNA', 'C680': 'CNA', 'C68A': 'CNA', 'C700': 'CNA',
    'C750': 'CNA', 'CRJ2': 'CR2', 'CRJ7': 'CR7', 'CRJ9': 'CR9', 'CRJX': 'CR9',
    'DH8C': 'DH8', 'DH8D': 'DH8', 'E120': 'EM2', 'E121': 'EM2', 'E135': 'ERJ',
    'E145': 'ER4', 'E170': 'E70', 'E190': 'E90', 'E195': 'E95', 'E290': 'E90',
    'E295': 'E95', 'E35L': 'ER4', 'E390': 'E90', 'E75L': 'E75', 'E75S': 'E75',
    'IL62': 'IL9', 'IL76': 'IL9', 'RJ70': 'AR1', 'SB20': 'SF3', 'SF34': 'SF3',
    'T204': 'T20'
}


def load_flights(path: str | Path = DEFAULT_CSV) -> pd.DataFrame:
	"""Read flights.csv and return its contents as a DataFrame."""
	# Keep flight numbers as text so values such as "0111X" are not changed.
	flights = pd.read_csv(path, dtype={"flight_number": "string"})
	# Convert this column to numbers so comparisons and calculations work.
	flights["occurrences"] = pd.to_numeric(flights["occurrences"], errors="raise")

	# Add cleaning or calculated columns here. For example:
	# flights["route"] = flights["origin"] + "-" + flights["destination"]
	return flights

def get_airports() -> list[str]:
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT iata_code FROM airports")
            rows = cur.fetchall()
    return [row[0] for row in rows]

def get_planes() -> list[str]:
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT iata_code FROM planes")
            rows = cur.fetchall()
    return [row[0] for row in rows]



def validate_airport(airport_code: str) -> bool:
    airports_list: list[str] = get_airports()
    valid_airports: set[str] = set(airports_list)
    return airport_code.upper() in valid_airports

def remove_invalid_flights(flights: pd.DataFrame, path: str | Path) -> pd.DataFrame:
    valid_airports = set(get_airports())
    valid_aircraft = set(get_planes())
    cleaned = flights[
        flights["origin"].isin(valid_airports)
        & flights["destination"].isin(valid_airports)
        & (flights["origin"] != flights["destination"])
    ].copy()
    cleaned = flights[
        ~flights["aircraft_type"].isin(invalid_aircraft)
    ].copy()
    cleaned.to_csv(path, index=False)
    print(f"Removed {len(flights) - len(cleaned):,} invalid flights. {len(cleaned):,} rows written to {path}.")
    return cleaned

def convert_aircraft_types(flights: pd.DataFrame, path: str | Path) -> pd.DataFrame:
    cleaned = flights.copy()
    converted_types = cleaned["aircraft_type"].map(ICAO_TO_IATA)
    converted_count = converted_types.notna().sum()
    cleaned["aircraft_type"] = (
        cleaned["aircraft_type"]
        .map(ICAO_TO_IATA)
        .fillna(cleaned["aircraft_type"])
    )
    print(f"Converted {converted_count:,} aircraft types")
    cleaned.to_csv(path, index=False)
    return cleaned

def print_aircraft_type(flights: pd.DataFrame) -> None:
    aircraft_list = set(flights["aircraft_type"].dropna())
    print(aircraft_list, len(aircraft_list))
    
def print_valid_aircraft() -> set[str]:
    print(get_planes())
        

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
	return (
		flights.groupby(["origin", "destination"], as_index=False)
		.agg(flight_count=("flight_number", "nunique"), occurrences=("occurrences", "sum"))
		.sort_values("occurrences", ascending=False)
	)
 
def remove_same_airport_flights(flights: pd.DataFrame, path: str | Path) -> pd.DataFrame:
    cleaned = flights.loc[flights["origin"] != flights["destination"]].copy()
    cleaned.to_csv(path, index=False)
    return cleaned

def count_same_airport_flights(flights: pd.DataFrame) -> int:
    return len(flights.loc[
        flights["origin"] == flights["destination"]
    ])

def count_distinct_airports(flights: pd.DataFrame) -> None:
    airports = set(flights["origin"].dropna()) | set(flights["destination"].dropna())
    print("Number of airports: ", len(airports))



def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	# Add command-line options here when you want to control new operations.
	parser.add_argument("--file", type=Path, default=DEFAULT_CSV, help="CSV file to load")
	parser.add_argument("--origin", help="Filter by origin airport")
	parser.add_argument("--destination", help="Filter by destination airport")
	parser.add_argument("--aircraft-type", help="Filter by aircraft type")
	parser.add_argument("--min-occurrences", type=int, help="Keep flights with at least this many occurrences")
	parser.add_argument("--remove-same-airport", action="store_true", help="Remove same-airport flights and overwrite the input file")
	parser.add_argument("--summary", action="store_true", help="Show totals grouped by route")
	parser.add_argument("--limit", type=int, help="Show only the first N matching rows")
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

    print(f"Loaded {len(flights):,} matching flights")
    same_count = count_same_airport_flights(flights)
    print(f"Same-airport flights: {same_count:,}")
    flights = remove_invalid_flights(flights, args.file)
    convert_aircraft_types(flights, args.file)
    count_distinct_airports(flights)


    if args.remove_same_airport:
        flights = remove_same_airport_flights(flights, args.file)
        print(f"Removed {same_count:,} same-airport flights. {len(flights):,} rows written to {args.file}")

    if args.summary:
        print(summarise_flights(flights).to_string(index=False))
        
            
        return

    if args.limit is not None:
        flights = flights.head(args.limit)

    #print(flights.to_string(index=False))
    



if __name__ == "__main__":
	main()

"""
Print a table of submission scores from the downloaded results/ JSON files.

Run download_results.py first to fetch the files. Rows are sorted by version.

Usage:
    python show_results.py
"""

import json
import re

import pandas as pd

from submission_bucket import RESULTS_DIR

VERSION_RE = re.compile(r"_v(\d+)\.parquet_result\.json$")


def load_results() -> pd.DataFrame:
    """Read every result JSON in RESULTS_DIR into a version-sorted DataFrame."""
    rows = []
    for path in RESULTS_DIR.glob(f"*{'_result.json'}"):
        match = VERSION_RE.search(path.name)
        if not match:
            continue
        data = json.loads(path.read_text())
        rows.append(
            {
                "version": int(match.group(1)),
                "score": data.get("score"),
                "status": data.get("status"),
                "used_pairs": data.get("used_pairs"),
            }
        )
    return pd.DataFrame(rows).sort_values("version").reset_index(drop=True)


def main() -> None:
    results = load_results()
    if results.empty:
        print(f"No result files in {RESULTS_DIR}. Run download_results.py first.")
        return
    print(results.to_string(index=False))
    best = results.loc[results["score"].idxmin()]
    print(f"\nBest: v{int(best['version'])} at {best['score']:.4f}")


if __name__ == "__main__":
    main()

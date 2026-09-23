"""
Download all grader result files from the submission bucket into results/.

Each submission produces a <file>.parquet_result.json object holding its score.
This downloads every such object so show_results.py can tabulate them.

Usage:
    python download_results.py
"""

from submission_bucket import BUCKET, RESULTS_DIR, make_client

RESULT_SUFFIX = "_result.json"


def download_results() -> None:
    """Download every *_result.json object from the bucket into RESULTS_DIR."""
    client = make_client()
    RESULTS_DIR.mkdir(exist_ok=True)
    count = 0
    for obj in client.list_objects(BUCKET):
        name = obj.object_name
        if not name.endswith(RESULT_SUFFIX):
            continue
        client.fget_object(BUCKET, name, str(RESULTS_DIR / name))
        count += 1
    print(f"Downloaded {count} result file(s) to {RESULTS_DIR}")


if __name__ == "__main__":
    download_results()

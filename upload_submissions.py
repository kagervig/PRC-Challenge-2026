"""
Upload a range of submission files to the competition MinIO bucket.

Usage:
    python upload_submissions.py <start> <end>          # inclusive range
    python upload_submissions.py <start> <end> --skip-existing

Uploads predictions/unique-umbrella_v<N>.parquet for every N in [start, end]
to the prc-2026-unique-umbrella bucket. Credentials are read from
credentials.json (never hard-code the secret key).

Re-uploading a version overwrites the existing object: MinIO PutObject has no
"already exists" guard, and the grader re-runs and overwrites the version's
_result.json. Use --skip-existing to leave already-uploaded versions untouched.
"""

import argparse

from submission_bucket import (
    BUCKET,
    PREDICTIONS_DIR,
    TEAM_NAME,
    make_client,
    object_exists,
)


def upload_range(start: int, end: int, skip_existing: bool) -> None:
    """Upload each version in [start, end] to the submission bucket."""
    client = make_client()
    for version in range(start, end + 1):
        filename = f"{TEAM_NAME}_v{version}.parquet"
        local_path = PREDICTIONS_DIR / filename
        if not local_path.exists():
            print(f"v{version}: SKIP — {local_path} not found")
            continue
        if skip_existing and object_exists(client, filename):
            print(f"v{version}: SKIP — already on bucket (--skip-existing)")
            continue
        if object_exists(client, filename):
            print(f"v{version}: overwriting existing object")
        client.fput_object(BUCKET, filename, str(local_path))
        print(f"v{version}: uploaded {filename}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("start", type=int, help="first version number (inclusive)")
    parser.add_argument("end", type=int, help="last version number (inclusive)")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="do not re-upload versions already present on the bucket",
    )
    args = parser.parse_args()
    if args.end < args.start:
        parser.error("end must be >= start")
    upload_range(args.start, args.end, args.skip_existing)


if __name__ == "__main__":
    main()

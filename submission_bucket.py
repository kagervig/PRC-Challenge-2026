"""Shared MinIO client, paths, and constants for the competition submission bucket."""

import json
from pathlib import Path

from minio import Minio
from minio.error import S3Error

DATA_DIR = Path(__file__).parent
PREDICTIONS_DIR = DATA_DIR / "predictions"
RESULTS_DIR = DATA_DIR / "results"
CREDENTIALS_FILE = DATA_DIR / "credentials.json"
ENDPOINT = "s3.opensky-network.org"
BUCKET = "prc-2026-unique-umbrella"
TEAM_NAME = "unique-umbrella"


def make_client() -> Minio:
    """Build a MinIO client from the service-account credentials file."""
    creds = json.loads(CREDENTIALS_FILE.read_text())
    return Minio(
        ENDPOINT,
        access_key=creds["accessKey"],
        secret_key=creds["secretKey"],
    )


def object_exists(client: Minio, object_name: str) -> bool:
    """Return True if object_name already exists in the bucket."""
    try:
        client.stat_object(BUCKET, object_name)
        return True
    except S3Error as err:
        if err.code in ("NoSuchKey", "NoSuchObject"):
            return False
        raise

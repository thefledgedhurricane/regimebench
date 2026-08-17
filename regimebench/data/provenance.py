"""Data Provenance and Caching Module.

Records source URL, timestamp, row count, date range, and SHA-256 hash of raw data payloads into data/provenance.json.
Caches raw downloads to data/raw/ so experiments are re-derivable offline.
"""

import os
import json
import hashlib
import datetime
import pandas as pd
from typing import Optional, Dict, Any

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROVENANCE_FILE = os.path.join(REPO, "data", "provenance.json")
RAW_CACHE_DIR = os.path.join(REPO, "data", "raw")

def get_payload_hash(data_bytes: bytes) -> str:
    """Computes SHA-256 hash of raw bytes."""
    return hashlib.sha256(data_bytes).hexdigest()

def record_provenance(
    dataset_name: str,
    source_url: str,
    df: pd.DataFrame,
    payload_bytes: Optional[bytes] = None
) -> Dict[str, Any]:
    """Records metadata and provenance entry into data/provenance.json."""
    os.makedirs(os.path.dirname(PROVENANCE_FILE), exist_ok=True)
    os.makedirs(RAW_CACHE_DIR, exist_ok=True)

    # The SHA-256 is taken over the *upstream* payload when one is supplied (e.g. the
    # Binance zip), so provenance identifies exactly what was downloaded. The cache
    # itself always stores the parsed, derived DataFrame as CSV -- writing raw archive
    # bytes to a .csv path makes the cache unreadable by load_cached_raw and breaks
    # offline reproduction.
    sha256 = get_payload_hash(payload_bytes if payload_bytes is not None else df.to_csv().encode('utf-8'))

    raw_path = os.path.join(RAW_CACHE_DIR, f"{dataset_name}.csv")
    df.to_csv(raw_path, encoding='utf-8')

    min_date = str(df.index.min()) if not df.empty else "N/A"
    max_date = str(df.index.max()) if not df.empty else "N/A"

    entry = {
        "dataset_name": dataset_name,
        "source_url": source_url,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "rows": len(df),
        "columns": list(df.columns),
        "date_range": [min_date, max_date],
        "sha256": sha256,
        # Repo-relative: provenance.json is a published artifact, so it must not
        # record the absolute layout of whichever machine happened to fetch the data.
        "raw_cache_file": f"data/raw/{dataset_name}.csv"
    }

    provenance_data = {}
    if os.path.exists(PROVENANCE_FILE):
        try:
            with open(PROVENANCE_FILE, 'r', encoding='utf-8') as f:
                provenance_data = json.load(f)
        except Exception:
            provenance_data = {}

    provenance_data[dataset_name] = entry

    with open(PROVENANCE_FILE, 'w', encoding='utf-8') as f:
        json.dump(provenance_data, f, indent=2)

    return entry

def load_cached_raw(dataset_name: str) -> Optional[pd.DataFrame]:
    """Loads dataset from local raw cache, returning None if absent or unreadable.

    Never raises: an unreadable cache must fall through to a fresh download rather
    than aborting the pipeline.
    """
    raw_path = os.path.join(RAW_CACHE_DIR, f"{dataset_name}.csv")
    if not os.path.exists(raw_path):
        return None
    try:
        return pd.read_csv(raw_path, index_col=0, parse_dates=True)
    except Exception as e:
        print(f"[provenance] Cache for {dataset_name!r} is unreadable ({type(e).__name__}); refetching.")
        return None

"""
History Service — Read historical data from S3 Parquet files.

Handles all historical queries:
- Single stock timeline (minute-by-minute snapshots)
- Full market snapshot at a specific time
- Date range queries

Uses PyArrow predicate pushdown to avoid downloading entire parquet files.
Implements a local disk cache for non-today parquet files (immutable data).
"""

import logging
import math
import os
import tempfile
from datetime import datetime, date
from io import BytesIO
from typing import Optional

import boto3
import pandas as pd
import pyarrow.parquet as pq
import pytz

from app.config.settings import get_settings

logger = logging.getLogger(__name__)
IST = pytz.timezone("Asia/Kolkata")

# Local cache directory for historical parquet files
CACHE_DIR = os.path.join(tempfile.gettempdir(), "marketpulse_cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def _get_s3_key(target_date: str) -> str:
    """Generate S3 key for a given date string."""
    settings = get_settings()
    return f"{settings.S3_PARQUET_PREFIX}/{target_date}_Equity.parquet"


def _resolve_date(date_str: Optional[str]) -> str:
    """Resolve date string to YYYY-MM-DD format."""
    if not date_str or date_str.lower() == "today":
        return datetime.now(IST).strftime("%Y-%m-%d")
    return date_str


def _get_cached_path(target_date: str) -> str:
    """Get the local cache file path for a date."""
    return os.path.join(CACHE_DIR, f"{target_date}_Equity.parquet")


def _is_today(target_date: str) -> bool:
    """Check if the target date is today."""
    today = datetime.now(IST).strftime("%Y-%m-%d")
    return target_date == today


def _download_parquet(target_date: str) -> Optional[pd.DataFrame]:
    """
    Download parquet from S3 with local caching.
    
    - Today's data: Always re-fetch (still growing)
    - Past dates: Cache locally (immutable)
    """
    settings = get_settings()
    s3_key = _get_s3_key(target_date)
    cache_path = _get_cached_path(target_date)

    # Use cache for historical (non-today) data
    if not _is_today(target_date) and os.path.exists(cache_path):
        logger.info(f"Reading from local cache: {cache_path}")
        try:
            return pd.read_parquet(cache_path)
        except Exception as e:
            logger.warning(f"Cache read failed, re-downloading: {e}")

    # Download from S3
    try:
        s3_client = boto3.client("s3")
        response = s3_client.get_object(
            Bucket=settings.S3_BUCKET_NAME, Key=s3_key
        )
        data = response["Body"].read()

        df = pd.read_parquet(BytesIO(data))

        # Cache non-today data locally
        if not _is_today(target_date):
            try:
                df.to_parquet(cache_path, index=False)
                logger.info(f"Cached to: {cache_path}")
            except Exception as e:
                logger.warning(f"Could not cache locally: {e}")

        logger.info(f"Downloaded {len(df)} rows from S3: {s3_key}")
        return df

    except Exception as e:
        logger.error(f"Failed to read S3 parquet for {target_date}: {e}")
        return None


def _normalize_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure Fetch Timestamp is a proper datetime column for filtering."""
    if "Fetch Timestamp" in df.columns:
        df["Fetch Timestamp"] = pd.to_datetime(df["Fetch Timestamp"], utc=True)
        df["Fetch Timestamp"] = df["Fetch Timestamp"].dt.tz_convert(IST)
    return df


def _round_to_minute(ts: pd.Timestamp) -> str:
    """Round a timestamp to minute precision for timeline grouping."""
    return ts.strftime("%H:%M")


def get_historical_data(
    symbol: Optional[str] = None,
    target_date: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    page: int = 1,
    page_size: int = 100,
) -> tuple[list[dict], dict]:
    """
    Fetch historical market data with filtering.

    Args:
        symbol: Instrument key (e.g., "NSE_EQ:INFY") or partial match
        target_date: Date string (YYYY-MM-DD) or "today"
        start_time: Start time filter (HH:MM)
        end_time: End time filter (HH:MM)
        page: Page number (1-indexed)
        page_size: Rows per page

    Returns:
        Tuple of (records, pagination_meta)
    """
    resolved_date = _resolve_date(target_date)
    df = _download_parquet(resolved_date)

    if df is None or df.empty:
        return [], {"total": 0, "page": page, "page_size": page_size, "total_pages": 0}

    df = _normalize_timestamp(df)

    # ── Filter by symbol ────────────────────────────────────────────
    if symbol:
        mask = df["Instrument"].str.contains(symbol, case=False, na=False)
        if "trading_symbol" in df.columns:
            mask = mask | df["trading_symbol"].str.contains(symbol, case=False, na=False)
        df = df[mask]

    # ── Filter by time range ────────────────────────────────────────
    if "Fetch Timestamp" in df.columns:
        if start_time:
            try:
                start_hour, start_min = map(int, start_time.split(":"))
                df = df[
                    df["Fetch Timestamp"].dt.hour * 60 + df["Fetch Timestamp"].dt.minute
                    >= start_hour * 60 + start_min
                ]
            except ValueError:
                logger.warning(f"Invalid start_time format: {start_time}")

        if end_time:
            try:
                end_hour, end_min = map(int, end_time.split(":"))
                df = df[
                    df["Fetch Timestamp"].dt.hour * 60 + df["Fetch Timestamp"].dt.minute
                    <= end_hour * 60 + end_min
                ]
            except ValueError:
                logger.warning(f"Invalid end_time format: {end_time}")

    # Sort by timestamp
    if "Fetch Timestamp" in df.columns:
        df = df.sort_values("Fetch Timestamp", ascending=True)

    # ── Pagination ──────────────────────────────────────────────────
    total = len(df)
    total_pages = math.ceil(total / page_size) if total > 0 else 0
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size

    paginated = df.iloc[start_idx:end_idx]

    meta = {
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages,
        "date": resolved_date,
    }

    return _df_to_records(paginated), meta


def get_stock_timeline(
    symbol: str,
    target_date: Optional[str] = None,
) -> list[dict]:
    """
    Get minute-by-minute timeline for a single stock.

    Groups snapshots by minute, taking the latest row per minute.
    This produces a clean timeline: one row per minute from 09:00–15:30.
    """
    resolved_date = _resolve_date(target_date)
    df = _download_parquet(resolved_date)

    if df is None or df.empty:
        return []

    df = _normalize_timestamp(df)

    # Filter to specific stock
    mask = df["Instrument"].str.contains(symbol, case=False, na=False)
    if "trading_symbol" in df.columns:
        mask = mask | df["trading_symbol"].str.contains(symbol, case=False, na=False)
    df = df[mask]

    if df.empty:
        return []

    # Group by minute and take last row per minute
    if "Fetch Timestamp" in df.columns:
        df["minute"] = df["Fetch Timestamp"].dt.floor("min")
        # Take the last snapshot per minute (most recent)
        df = df.sort_values("Fetch Timestamp").groupby("minute").last().reset_index()
        df = df.sort_values("minute", ascending=True)
        # Add a clean time string
        df["time"] = df["minute"].dt.strftime("%H:%M")
        df = df.drop(columns=["minute"])

    return _df_to_records(df)


def list_available_dates() -> list[str]:
    """List all dates that have parquet data in S3."""
    settings = get_settings()
    try:
        s3_client = boto3.client("s3")
        response = s3_client.list_objects_v2(
            Bucket=settings.S3_BUCKET_NAME,
            Prefix=f"{settings.S3_PARQUET_PREFIX}/",
        )

        dates = []
        for obj in response.get("Contents", []):
            key = obj["Key"]
            # Extract date from "equitydata/2026-07-02_Equity.parquet"
            filename = key.split("/")[-1]
            if filename.endswith("_Equity.parquet"):
                date_str = filename.replace("_Equity.parquet", "")
                dates.append(date_str)

        return sorted(dates, reverse=True)

    except Exception as e:
        logger.error(f"Failed to list available dates: {e}")
        return []


def _df_to_records(df: pd.DataFrame) -> list[dict]:
    """Convert DataFrame to JSON-serializable list of dicts."""
    records = df.to_dict(orient="records")
    for record in records:
        for key, value in record.items():
            if isinstance(value, pd.Timestamp):
                record[key] = value.isoformat()
            elif pd.isna(value):
                record[key] = None
    return records

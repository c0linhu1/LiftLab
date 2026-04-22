"""
clean.py — Clean and prepare raw session data for experiment assignment.

Taking the flattened GA4 BigQuery export and producing a standardized
sessions table 

"""
from pathlib import Path

import numpy as np
import pandas as pd


def load_raw_sessions(data_dir: str = "./data") -> pd.DataFrame: # including type hint for arg and return value
    """Load the raw BigQuery export CSV."""
    # converts string to path object - pointing to data folder and then appending to the filename
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "data"
    else:
        data_dir = Path(data_dir)
    path = data_dir / "sessions_raw.csv"

    if not path.exists():
        raise FileNotFoundError(f"Raw sessions file not found: {path}")

    return pd.read_csv(path)


def clean_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw session data.

    Steps:
    - Standardize column types
    - Handle nulls in revenue and event counts
    - Parse dates and numeric timestamps
    - Clean categorical fields
    - Create derived session-level features
    - Create leakage-safe historical user features for CUPED / segmentation

    """
    df = df.copy()

    required_cols = [
        "user_pseudo_id",
        "session_id",
        "event_date",
        "revenue",
        "transactions",
        "add_to_cart_events",
        "pageviews",
        "event_count",
        "session_start",
        "session_end",
        "device_type",
        "country",
        "traffic_source",
        "traffic_medium",
    ]
    # checking for missing cols
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # ID / date parsing
    df["user_pseudo_id"] = df["user_pseudo_id"].astype("string")
    df["session_id"] = df["session_id"].astype("string")
    df["event_date"] = pd.to_datetime(df["event_date"], format="%Y%m%d", errors="coerce")

    # numeric parsing
    numeric_cols = [
        "revenue",
        "transactions",
        "add_to_cart_events",
        "pageviews",
        "event_count",
        "session_start",
        "session_end",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # null handling

    # revenue nulls usually mean non-purchase sessions
    df["revenue"] = df["revenue"].fillna(0.0)

    # count/event fields default to 0 when missing
    df["transactions"] = df["transactions"].fillna(0).astype(int)
    df["add_to_cart_events"] = df["add_to_cart_events"].fillna(0).astype(int)
    df["pageviews"] = df["pageviews"].fillna(0).astype(int)
    df["event_count"] = df["event_count"].fillna(0).astype(int)


    # clean categorical fields - filling potential null values w 'unknown' and standardizing text
    df["device_type"] = df["device_type"].fillna("unknown").astype("string").str.lower().str.strip()
    df["country"] = df["country"].fillna("unknown").astype("string").str.upper().str.strip()
    df["traffic_source"] = df["traffic_source"].fillna("unknown").astype("string").str.lower().str.strip()
    df["traffic_medium"] = df["traffic_medium"].fillna("unknown").astype("string").str.lower().str.strip()
    

    # dedup - avoids potentiall double-counting 
    df = df.drop_duplicates(subset=["user_pseudo_id", "session_id"]).copy()


    # Derived session-level features
    # binary conversion flag
    df["converted"] = (df["transactions"] > 0).astype(int)

    # get the session duration in seconds bc GA4 timestamps assumed microseconds - clip negative durations to 0
    df["session_duration_sec"] = ((df["session_end"] - df["session_start"]) / 1_000_000).clip(lower=0)

    # if duration is neg replace w 0
    df["session_duration_sec"] = df["session_duration_sec"].fillna(0)

    # bounce proxy: low page depth + very short session
    df["is_bounce_proxy"] = (
        (df["pageviews"] <= 1) & (df["session_duration_sec"] < 10)
    ).astype(int)   

    # engagement score: composite metric for CUPED / segmentation
    # standardizing w log transform to handle skew from duration
    df["engagement_score"] = (
        np.log1p(df["pageviews"]) * 0.4   # pageviews have highest weight because bigger sign of engagement
        + np.log1p(df["event_count"]) * 0.3
        + np.log1p(df["session_duration_sec"]) * 0.3
    )


    # leakage-safe historical user features

    # Sort so "prior" truly means earlier sessions for each user
    df = df.sort_values(["user_pseudo_id", "session_start", "session_id"]).reset_index(drop=True)

    # # of sessions before the current one
    df["prior_sessions"] = df.groupby("user_pseudo_id").cumcount()

    # cumulative historical sums excluding the current row
    df["prior_pageviews"] = (
        df.groupby("user_pseudo_id")["pageviews"].cumsum() - df["pageviews"]
    )

    df["prior_revenue"] = (
        df.groupby("user_pseudo_id")["revenue"].cumsum() - df["revenue"]
    )

    df["prior_transactions"] = (
        df.groupby("user_pseudo_id")["transactions"].cumsum() - df["transactions"]
    )

    # Historical average engagement excluding current session
    df["prior_avg_engagement"] = (
        df.groupby("user_pseudo_id")["engagement_score"]
        .transform(lambda s: s.shift(1).expanding().mean())
        .fillna(0)
    )

    return df


def build_clean_sessions(data_dir: str = "./data") -> pd.DataFrame:
    """loading and cleaning sessions"""
    raw = load_raw_sessions(data_dir)
    clean = clean_sessions(raw)

    return clean


if __name__ == "__main__":
    df = build_clean_sessions()

    print(f"Cleaned sessions: {len(df):,} rows")
    print(f"Unique users: {df['user_pseudo_id'].nunique():,}")
    print(f"Date range: {df['event_date'].min()} to {df['event_date'].max()}")
    print(f"Conversion rate: {df['converted'].mean():.4f}")
    print(f"Avg session duration: {df['session_duration_sec'].mean():.1f}s")
    print(f"Bounce proxy rate: {df['is_bounce_proxy'].mean():.4f}")

    print("\nDevice breakdown:")
    print(df["device_type"].value_counts(dropna=False))

    print("\nRevenue stats (purchasers only):")
    purchasers = df[df["revenue"] > 0]["revenue"]
    if len(purchasers) > 0:
        print(purchasers.describe())
    else:
        print("No purchasing sessions found.")


    # convert to csv
    output_path = Path("./data/sessions_clean.csv")
    df.to_csv(output_path, index=False)
    print(f"\nSaved cleaned sessions to {output_path} ({output_path.stat().st_size / 1_000_000:.1f} MB)")

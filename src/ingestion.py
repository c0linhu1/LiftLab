"""
ingest.py — BigQuery ingestion script.

Pulls flattened session data from the GA4 public dataset and saves
it locally as CSV for development. This is the "production" ingestion
path — run it once to generate data/sessions_raw.csv, or on a schedule
if the source data updates.

Usage:
    python src/ingest.py
    python src/ingest.py --start-date 20201101 --end-date 20210131 --output data/sessions_raw.csv

Requires:
    - google-cloud-bigquery installed
    - GCP credentials configured (service account key or gcloud auth)
    - Set GOOGLE_APPLICATION_CREDENTIALS env var, or run `gcloud auth application-default login`
"""
import pandas as pd
import argparse
from pathlib import Path

try:
    from google.cloud import bigquery
except ImportError:
    bigquery = None


FLATTEN_QUERY = """
SELECT
  user_pseudo_id,
  (SELECT value.int_value FROM UNNEST(event_params) WHERE key = 'ga_session_id') AS session_id,
  event_date,
  device.category AS device_type,
  geo.country,
  traffic_source.source AS traffic_source,
  traffic_source.medium AS traffic_medium,
  COUNT(*) AS event_count,
  COUNTIF(event_name = 'page_view') AS pageviews,
  COUNTIF(event_name = 'purchase') AS transactions,
  SUM(ecommerce.purchase_revenue) AS revenue,
  COUNTIF(event_name = 'add_to_cart') AS add_to_cart_events,
  MIN(event_timestamp) AS session_start,
  MAX(event_timestamp) AS session_end
FROM `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*`
WHERE _TABLE_SUFFIX BETWEEN @start_date AND @end_date
GROUP BY
  user_pseudo_id, session_id, event_date,
  device_type, country, traffic_source, traffic_medium
ORDER BY event_date
"""


def pull_from_bigquery(
    start_date: str = "20201101",
    end_date: str = "20210131",
    project_id: str = "liftlab-493800",
) -> pd.DataFrame:
    """
    Query BigQuery and return a pandas DataFrame.
    Requires google-cloud-bigquery and valid GCP credentials.
    """
    if bigquery is None:
        raise ImportError(
            "google-cloud-bigquery is not installed. "
            "Run: pip install google-cloud-bigquery db-dtypes"
        )

    client = bigquery.Client(project=project_id)

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("start_date", "STRING", start_date),
            bigquery.ScalarQueryParameter("end_date", "STRING", end_date),
        ]
    )

    print(f"Querying BigQuery for {start_date} to {end_date}...")
    df = client.query(FLATTEN_QUERY, job_config=job_config).to_dataframe()
    print(f"Pulled {len(df):,} rows, {df['user_pseudo_id'].nunique():,} unique users")

    return df


def save_local(df, output_path: str = "./data/sessions_raw.csv"):
    """Save DataFrame to local CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved to {path} ({path.stat().st_size / 1_000_000:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pull session data from BigQuery")
    parser.add_argument("--start-date", default="20201101", help="Start date (YYYYMMDD)")
    parser.add_argument("--end-date", default="20210131", help="End date (YYYYMMDD)")
    parser.add_argument("--output", default="./data/sessions_raw.csv", help="Output file path")
    parser.add_argument("--project-id", default="liftlab-493800", help="GCP project ID")
    args = parser.parse_args()

    df = pull_from_bigquery(args.start_date, args.end_date, args.project_id)
    save_local(df, args.output)
-- flatten_sessions.sql
-- Flattens nested GA4 event-level data into session-level records.
-- Run against: bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*
-- Output: one row per user-session with aggregated metrics.

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
GROUP BY
  user_pseudo_id,
  session_id,
  event_date,
  device_type,
  country,
  traffic_source,
  traffic_medium
ORDER BY event_date
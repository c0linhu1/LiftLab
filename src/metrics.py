"""
metrics.py — Experiment metrics computation.

Computes primary, secondary, and guardrail metrics for each variant.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass


@dataclass
class MetricResult:
    """Result for a single metric comparison."""
    metric_name: str
    control_mean: float
    treatment_mean: float
    absolute_lift: float
    relative_lift: float
    control_n: int
    treatment_n: int

# internal function
def _compute_metric(df: pd.DataFrame, column: str, metric_name: str) -> MetricResult:
    """Compute a single metric comparison between variants."""
    # isolating control and treatment groups for the specified metric column
    ctrl = df[df["variant"] == "control"][column]
    treat = df[df["variant"] == "treatment"][column]

    ctrl_mean = ctrl.mean()
    treat_mean = treat.mean()

    # difference between control and treatment means, both absolute and relative
    abs_lift = treat_mean - ctrl_mean
    rel_lift = abs_lift / ctrl_mean if ctrl_mean != 0 else 0.0

    # returning MetricResult instance
    return MetricResult(
        metric_name=metric_name,
        control_mean=ctrl_mean,
        treatment_mean=treat_mean,
        absolute_lift=abs_lift,
        relative_lift=rel_lift,
        control_n=len(ctrl),
        treatment_n=len(treat),
    )


def compute_primary_metrics(df: pd.DataFrame) -> list[MetricResult]:
    """Primary metrics - conversion rate and revenue per session."""
    return [
        _compute_metric(df, "converted", "conversion_rate"),
        _compute_metric(df, "revenue", "revenue_per_session"),
    ]


def compute_secondary_metrics(df: pd.DataFrame) -> list[MetricResult]:
    """Secondary metrics — not as important as conversion/revenue but still relevant to the experiment's hypothesis"""
    return [
        _compute_metric(df, "add_to_cart_events", "add_to_cart_rate"),
        _compute_metric(df, "pageviews", "avg_pageviews"),
        _compute_metric(df, "session_duration_sec", "avg_session_duration"),
        _compute_metric(df, "engagement_score", "avg_engagement_score"),
    ]


def compute_guardrail_metrics(df: pd.DataFrame) -> list[MetricResult]:
    """Guardrail metrics — these must not regress."""
    metrics = [
        _compute_metric(df, "is_bounce_proxy", "bounce_rate"),
        _compute_metric(df, "pageviews", "session_depth"),
    ]


    # creating new guardrail metric - revenue volatility - higher means more unpredictable revenue -> bad
    # revenue volatility: std of revenue among purchasers only
    purchasers = df[df["revenue"] > 0]
    ctrl_rev = purchasers[purchasers["variant"] == "control"]["revenue"]
    treat_rev = purchasers[purchasers["variant"] == "treatment"]["revenue"]

    if len(ctrl_rev) > 0 and len(treat_rev) > 0:
        ctrl_std = ctrl_rev.std()
        treat_std = treat_rev.std()
        metrics.append(MetricResult(
            metric_name="revenue_volatility",
            control_mean=ctrl_std,
            treatment_mean=treat_std,
            absolute_lift=treat_std - ctrl_std,
            relative_lift=(treat_std - ctrl_std) / ctrl_std if ctrl_std > 0 else 0.0,
            control_n=len(ctrl_rev),
            treatment_n=len(treat_rev),
        ))

    return metrics

def _to_dict(m: MetricResult) -> dict:
    """Convert MetricResult to JSON-serializable dict."""
    return {
        "metric_name": m.metric_name,
        "control_mean": round(m.control_mean, 6),
        "treatment_mean": round(m.treatment_mean, 6),
        "absolute_lift": round(m.absolute_lift, 6),
        "relative_lift": round(m.relative_lift, 6),
        "control_n": m.control_n,
        "treatment_n": m.treatment_n,
    }


def compute_all_metrics(df: pd.DataFrame) -> dict:
    """Compute all metric tiers and return as structured dict."""
    return {
        "primary": [_to_dict(m) for m in compute_primary_metrics(df)],
        "secondary": [_to_dict(m) for m in compute_secondary_metrics(df)],
        "guardrail": [_to_dict(m) for m in compute_guardrail_metrics(df)],
    }




if __name__ == "__main__":
    from clean import build_clean_sessions
    from assign_experiment import assign_users
    from simulate_treatment import simulate_treatment_effects

    df = build_clean_sessions()
    df = assign_users(df)
    df = simulate_treatment_effects(df)

    results = compute_all_metrics(df)

    print("PRIMARY METRICS:")
    for m in results["primary"]:
        print(f"{m['metric_name']}: control={m['control_mean']:.4f}, "
              f"treatment={m['treatment_mean']:.4f}, "
              f"lift={m['relative_lift']:.2%}")

    print("\nSECONDARY METRICS")
    for m in results["secondary"]:
        print(f"{m['metric_name']}: control={m['control_mean']:.4f}, "
              f"treatment={m['treatment_mean']:.4f}, "
              f"lift={m['relative_lift']:.2%}")

    print("\nGUARDRAIL METRICS")
    for m in results["guardrail"]:
        print(f"{m['metric_name']}: control={m['control_mean']:.4f}, "
              f"treatment={m['treatment_mean']:.4f}, "
              f"lift={m['relative_lift']:.2%}")
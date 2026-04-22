"""
assign_experiment.py — Experiment assignment engine.

Assigns users to control/treatment groups with proper randomization.
Supports stratified assignment by device type.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment - including type hints"""
    experiment_id: str
    experiment_name: str
    hypothesis: str
    primary_metric: str
    guardrail_metrics: list[str] = field(default_factory=list)
    treatment_split: float = 0.5
    stratify_by: Optional[list[str]] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None  
    seed: int = 42


# Default experiment config
DEFAULT_EXPERIMENT = ExperimentConfig(
    experiment_id="exp_001",
    experiment_name="Checkout Flow Redesign",
    hypothesis=(
        "A streamlined checkout flow will increase conversion rate "
        "and revenue per session without degrading engagement metrics."
    ),
    primary_metric="conversion_rate",
    guardrail_metrics=["bounce_rate", "session_depth", "revenue_volatility"],
    treatment_split=0.5,
    stratify_by=["device_type"], # ensuring balanced assignment across mainly mobile/desktop
    start_date="2020-12-01",
    end_date="2021-01-31",
    seed=42,
)


def assign_users(
    df: pd.DataFrame,
    config: ExperimentConfig = DEFAULT_EXPERIMENT,
) -> pd.DataFrame:
    """
    Assign users to experiment variants.

    Assignment is at the USER level — a user sees the same variant
    across all their sessions. Uses seeded randomization for
    reproducibility.
    """
    df = df.copy()

    # filter to experiment date range
    if config.start_date:
        df = df[df["event_date"] >= pd.to_datetime(config.start_date)]
    if config.end_date:
        df = df[df["event_date"] <= pd.to_datetime(config.end_date)]

    users = df["user_pseudo_id"].unique()
    rng = np.random.RandomState(config.seed)

    if config.stratify_by:
        assignments = _stratified_assign(df, config, rng)
    else:
        assignments = _simple_assign(users, config, rng)

    # Merge assignments back to session-level data
    assignment_df = pd.DataFrame({
        "user_pseudo_id": list(assignments.keys()),
        "variant": list(assignments.values()),
        "experiment_id": config.experiment_id,
    })

    df = df.merge(assignment_df, on="user_pseudo_id", how="inner")
    df["experiment_name"] = config.experiment_name

    return df


def _simple_assign(
    users: np.ndarray,
    config: ExperimentConfig,
    rng: np.random.RandomState,
) -> dict:
    """Simple random assignment."""
    n_treatment = int(len(users) * config.treatment_split)
    shuffled = rng.permutation(users)
    return {
        user: "treatment" if i < n_treatment else "control"
        for i, user in enumerate(shuffled)
    }


def _stratified_assign(
    df: pd.DataFrame,
    config: ExperimentConfig,
    rng: np.random.RandomState,
) -> dict:
    """
    Stratified random assignment.
    Ensures balanced variant split within each stratum.
    Uses each user's most common value for the stratification column.
    """
    strat_cols = config.stratify_by

    # Get dominant stratum for each user (e.g., their most common device)
    user_strata = (
        df.groupby("user_pseudo_id")[strat_cols]
        .agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "unknown")
        .reset_index()
    )

    assignments = {}
    for _, group in user_strata.groupby(strat_cols):
        group_users = group["user_pseudo_id"].values
        n_treatment = int(len(group_users) * config.treatment_split)
        shuffled = rng.permutation(group_users)
        for i, user in enumerate(shuffled):
            assignments[user] = "treatment" if i < n_treatment else "control"

    return assignments


def get_assignment_summary(df: pd.DataFrame) -> dict:
    """Summarize experiment assignment balance."""
    return {
        "total_users": df["user_pseudo_id"].nunique(),
        "total_sessions": len(df),
        "variant_users": df.groupby("variant")["user_pseudo_id"].nunique().to_dict(),
        "variant_sessions": df.groupby("variant").size().to_dict(),
        "date_range": {
            "start": str(df["event_date"].min().date()),
            "end": str(df["event_date"].max().date()),
        },
    }


if __name__ == "__main__":
    from clean import build_clean_sessions

    df = build_clean_sessions()
    assigned = assign_users(df)
    summary = get_assignment_summary(assigned)

    print(f"Total users: {summary['total_users']:,}")
    print(f"Total sessions: {summary['total_sessions']:,}")
    print(f"Date range: {summary['date_range']}")
    print(f"\nVariant split (users): {summary['variant_users']}")
    print(f"Variant split (sessions): {summary['variant_sessions']}")
    print(f"\nBalance by device:")
    print(pd.crosstab(assigned["device_type"], assigned["variant"], normalize="index"))
"""
conftest.py — Pytest shared setup.

Adds src/ to sys.path so tests can `from frequentist import ...` etc.
without needing PYTHONPATH=/app like the API uses. Also provides
small synthetic-data fixtures shared across test files.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


@pytest.fixture
def synthetic_user_df():
    """
    Tiny user-level DataFrame for fast inference tests.

    Control: 10% conversion, mean revenue 5 (only converters spend).
    Treatment: 20% conversion, mean revenue 6.
    Other columns held constant so any single-metric test is isolated.
    """
    n_per_arm = 500
    rng = np.random.default_rng(7)

    def arm(name: str, conv_rate: float, rev_mean: float):
        converted = (rng.random(n_per_arm) < conv_rate).astype(int)
        revenue = np.where(converted == 1, rng.normal(rev_mean, 1.0, n_per_arm), 0.0).clip(min=0)
        return pd.DataFrame({
            "user_pseudo_id": [f"{name}_{i}" for i in range(n_per_arm)],
            "variant": name,
            "converted": converted,
            "transactions": converted,
            "revenue": revenue,
            "add_to_cart_events": rng.poisson(2, n_per_arm),
            "pageviews": rng.poisson(5, n_per_arm),
            "event_count": rng.poisson(10, n_per_arm),
            "session_duration_sec": rng.normal(120, 30, n_per_arm).clip(min=0),
            "engagement_score": rng.normal(1.5, 0.3, n_per_arm),
            "is_bounce_proxy": (rng.random(n_per_arm) < 0.25).astype(int),
            "prior_sessions": rng.integers(0, 5, n_per_arm),
            "prior_revenue": rng.exponential(2, n_per_arm),
            "prior_avg_engagement": rng.normal(1.2, 0.4, n_per_arm),
        })

    return pd.concat(
        [arm("control", 0.10, 5.0), arm("treatment", 0.20, 6.0)],
        ignore_index=True,
    )

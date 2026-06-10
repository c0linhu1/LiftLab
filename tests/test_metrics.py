"""
test_metrics.py — Sanity checks for src/metrics.py.

Validates:
- lift math (absolute = treat_mean - ctrl_mean, relative = abs / ctrl_mean)
- three-tier structure of compute_all_metrics
- revenue_volatility (std of revenue among PURCHASERS only)
- compute_all_metrics returns serializable dicts
"""

import numpy as np
import pandas as pd
import pytest

from metrics import (
    MetricResult,
    compute_primary_metrics,
    compute_secondary_metrics,
    compute_guardrail_metrics,
    compute_all_metrics,
)


def _find(metrics_list, name):
    for m in metrics_list:
        if (m.metric_name if isinstance(m, MetricResult) else m["metric_name"]) == name:
            return m
    raise AssertionError(f"metric {name!r} not found")


def test_lift_math_is_correct(synthetic_user_df):
    """Absolute lift = treat - ctrl; relative = abs / ctrl."""
    primary = compute_primary_metrics(synthetic_user_df)
    conv = _find(primary, "conversion_rate")
    expected_abs = conv.treatment_mean - conv.control_mean
    expected_rel = expected_abs / conv.control_mean
    assert conv.absolute_lift == pytest.approx(expected_abs)
    assert conv.relative_lift == pytest.approx(expected_rel)


def test_conversion_lift_matches_construction(synthetic_user_df):
    """Fixture has 10% vs 20% conversion → lift ≈ +100%, give or take noise."""
    primary = compute_primary_metrics(synthetic_user_df)
    conv = _find(primary, "conversion_rate")
    assert conv.control_mean == pytest.approx(0.10, abs=0.04)
    assert conv.treatment_mean == pytest.approx(0.20, abs=0.04)
    assert conv.relative_lift > 0.6     # ~100% but with noise allowed


def test_three_tier_structure(synthetic_user_df):
    """compute_all_metrics returns dict with primary/secondary/guardrail keys."""
    all_m = compute_all_metrics(synthetic_user_df)
    assert set(all_m.keys()) == {"primary", "secondary", "guardrail"}
    for tier in all_m.values():
        assert isinstance(tier, list) and len(tier) > 0
        for m in tier:
            assert {"metric_name", "control_mean", "treatment_mean",
                    "absolute_lift", "relative_lift",
                    "control_n", "treatment_n"} <= set(m)


def test_revenue_volatility_uses_purchasers_only(synthetic_user_df):
    """Std should be of revenue WHERE revenue > 0, not over all rows."""
    guards = compute_guardrail_metrics(synthetic_user_df)
    vol = _find(guards, "revenue_volatility")
    purchasers = synthetic_user_df[synthetic_user_df["revenue"] > 0]
    ctrl_purchaser_std = purchasers[purchasers["variant"] == "control"]["revenue"].std()
    assert vol.control_mean == pytest.approx(ctrl_purchaser_std, rel=1e-6)


def test_results_are_json_serializable(synthetic_user_df):
    """compute_all_metrics returns plain dicts (not dataclasses)."""
    import json
    all_m = compute_all_metrics(synthetic_user_df)
    json.dumps(all_m)  # should not raise

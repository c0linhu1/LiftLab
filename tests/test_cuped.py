"""
test_cuped.py — Correctness checks for src/cuped.py.

Validates the two non-negotiable CUPED properties:
1. Strong Y~X correlation should produce meaningful variance reduction.
2. CUPED should NOT bias the treatment-control difference (it just
   reduces noise around the estimate).

Plus edge cases — zero-variance covariate, no correlation.
"""

import numpy as np
import pandas as pd
import pytest

from cuped import (
    _cuped_theta,
    cuped_adjust,
    run_cuped_test,
    aggregate_with_covariates,
)


def test_uncorrelated_covariate_gives_near_zero_theta():
    """When X has no relationship to Y, OLS coefficients ≈ 0."""
    rng = np.random.default_rng(42)
    n = 2000
    y = rng.normal(10, 2, n)
    X = rng.normal(0, 1, (n, 3))

    theta = _cuped_theta(y, X)
    assert np.abs(theta).max() < 0.2


def test_strong_correlation_reduces_variance_substantially():
    """Y = 2X + small noise → CUPED should remove most of Y's variance."""
    rng = np.random.default_rng(42)
    n = 2000
    x = rng.normal(0, 1, n)
    y = 2 * x + rng.normal(0, 0.5, n)

    y_adj, theta = cuped_adjust(y, x.reshape(-1, 1))

    assert theta[0] == pytest.approx(2.0, abs=0.1)
    var_reduction = 1 - y_adj.var() / y.var()
    assert var_reduction > 0.7   # most of variance removed


def test_cuped_does_not_bias_treatment_effect():
    """
    Y depends on X but treatment effect is real and constant.
    Both raw and CUPED-adjusted estimators should recover the same lift
    (CUPED just narrows the SE; it doesn't shift the estimate).
    """
    rng = np.random.default_rng(42)
    n_per = 5000
    true_effect = 1.0

    x_ctrl = rng.normal(0, 1, n_per)
    x_treat = rng.normal(0, 1, n_per)
    y_ctrl = 2 * x_ctrl + rng.normal(0, 1, n_per)
    y_treat = 2 * x_treat + true_effect + rng.normal(0, 1, n_per)

    y = np.concatenate([y_ctrl, y_treat])
    X = np.concatenate([x_ctrl, x_treat]).reshape(-1, 1)
    variant = np.array(["control"] * n_per + ["treatment"] * n_per)

    y_adj, _ = cuped_adjust(y, X)

    raw_diff = y[variant == "treatment"].mean() - y[variant == "control"].mean()
    adj_diff = y_adj[variant == "treatment"].mean() - y_adj[variant == "control"].mean()

    # Both should land near the true effect
    assert raw_diff == pytest.approx(true_effect, abs=0.1)
    assert adj_diff == pytest.approx(true_effect, abs=0.1)
    # And critically, the two estimates should agree
    assert adj_diff == pytest.approx(raw_diff, abs=0.05)


def test_zero_variance_covariate_does_not_crash():
    """A constant covariate should be silently ignored (theta ≈ 0)."""
    rng = np.random.default_rng(42)
    n = 500
    y = rng.normal(0, 1, n)
    X = np.column_stack([rng.normal(0, 1, n), np.ones(n)])  # second col is constant

    y_adj, theta = cuped_adjust(y, X)

    assert np.isfinite(theta).all()
    assert np.isfinite(y_adj).all()
    # The constant covariate should contribute ~nothing
    assert abs(theta[1]) < 1e-8


def test_run_cuped_test_returns_expected_fields(synthetic_user_df):
    """End-to-end check on the public API."""
    df = synthetic_user_df.copy()
    result = run_cuped_test(
        df, "revenue", "test_metric",
        covariates=["prior_sessions", "prior_revenue", "prior_avg_engagement"],
    )
    assert result.metric_name == "test_metric"
    assert isinstance(result.variance_reduction, float)
    assert isinstance(result.ci_width_improvement, float)
    assert len(result.theta) == 3
    assert result.control_n + result.treatment_n == len(df)


def test_variance_reduction_is_nonnegative_in_expectation(synthetic_user_df):
    """
    CUPED variance can technically be slightly higher than raw on a single
    sample (small-n estimation noise of theta), but in expectation it
    shouldn't go negative for any reasonable signal. Use a relaxed bound.
    """
    result = run_cuped_test(
        synthetic_user_df, "revenue", "rev",
        covariates=["prior_sessions", "prior_revenue", "prior_avg_engagement"],
    )
    assert result.variance_reduction > -0.05

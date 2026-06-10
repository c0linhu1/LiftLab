"""
test_bayesian.py — Correctness checks for src/bayesian.py.

Validates:
- Beta posterior behavior at boundary cases (no effect → P(T>C) ≈ 0.5,
  strong effect → P(T>C) ≈ 1)
- Loss calculation symmetry
- Normal posterior on the mean produces reasonable credible intervals
- run_bayesian_test dispatches binary vs continuous correctly
"""

import numpy as np
import pandas as pd
import pytest

from bayesian import (
    _sample_beta_posterior,
    _sample_normal_posterior,
    _compute_loss_and_prob,
    run_bayesian_test,
)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def test_beta_posterior_no_effect_gives_50_50(rng):
    """Identical successes/trials → P(T > C) ≈ 0.5."""
    n_samples = 50_000
    ctrl = _sample_beta_posterior(100, 1000, n_samples, rng)
    treat = _sample_beta_posterior(100, 1000, n_samples, rng)
    prob = (treat > ctrl).mean()
    assert 0.45 < prob < 0.55


def test_beta_posterior_strong_effect_gives_near_certainty(rng):
    """10% vs 20% with n=1000 each → P(T > C) > 0.99."""
    n_samples = 50_000
    ctrl = _sample_beta_posterior(100, 1000, n_samples, rng)
    treat = _sample_beta_posterior(200, 1000, n_samples, rng)
    prob = (treat > ctrl).mean()
    assert prob > 0.99


def test_normal_posterior_mean_matches_data_mean(rng):
    """Posterior mean should sit on x̄."""
    samples = _sample_normal_posterior(mean=10.0, var=4.0, n=400, n_samples=50_000, rng=rng)
    assert samples.mean() == pytest.approx(10.0, abs=0.05)


def test_normal_posterior_se_matches_formula(rng):
    """Posterior std should be √(var/n)."""
    samples = _sample_normal_posterior(mean=0.0, var=4.0, n=100, n_samples=50_000, rng=rng)
    expected_se = np.sqrt(4.0 / 100)  # = 0.2
    assert samples.std() == pytest.approx(expected_se, rel=0.05)


def test_loss_symmetric_for_identical_distributions(rng):
    """When distributions are identical, ship loss ≈ hold loss."""
    samples_a = rng.normal(0, 1, 50_000)
    samples_b = rng.normal(0, 1, 50_000)
    prob, _, _, loss_ship, loss_no_ship = _compute_loss_and_prob(samples_a, samples_b)
    assert 0.45 < prob < 0.55
    assert loss_ship == pytest.approx(loss_no_ship, abs=0.02)


def test_loss_asymmetric_when_treatment_wins(rng):
    """When T clearly > C, hold loss should dwarf ship loss."""
    ctrl = rng.normal(0, 1, 50_000)
    treat = rng.normal(2, 1, 50_000)
    _, _, _, loss_ship, loss_no_ship = _compute_loss_and_prob(ctrl, treat)
    assert loss_no_ship > 10 * loss_ship


def test_run_bayesian_test_binary_dispatch(synthetic_user_df):
    """Binary column → beta_binomial test."""
    r = run_bayesian_test(synthetic_user_df, "converted", "conv", n_samples=20_000)
    assert r.test_type == "beta_binomial"
    # Synthetic fixture is 10% vs 20% with n=500/arm; should be confident
    assert r.prob_treatment_better > 0.95


def test_run_bayesian_test_continuous_dispatch(synthetic_user_df):
    """Continuous column → normal_normal test."""
    r = run_bayesian_test(synthetic_user_df, "revenue", "rev", n_samples=20_000)
    assert r.test_type == "normal_normal"
    assert r.treatment_mean > r.control_mean      # fixture has treatment higher
    assert isinstance(r.credible_low, float)
    assert isinstance(r.credible_high, float)
    assert r.credible_low < r.credible_high


def test_credible_interval_brackets_observed_lift(synthetic_user_df):
    """The point estimate should sit inside its own 95% CrI (almost always)."""
    r = run_bayesian_test(synthetic_user_df, "revenue", "rev", n_samples=20_000)
    assert r.credible_low <= r.absolute_lift <= r.credible_high

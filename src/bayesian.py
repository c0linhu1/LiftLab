"""
bayesian.py — Bayesian A/B testing for experiment metrics.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass

from frequentist import USER_AGG, aggregate_to_user_level


N_SAMPLES = 100_000
SEED = 42

# Decision thresholds for ship_recommended
PROB_SHIP_THRESHOLD = 0.95     # P(treat > ctrl) required to ship
LOSS_SHIP_THRESHOLD_REL = 0.005  # expected ship-loss must be < this fraction of |ctrl_mean|


@dataclass
class BayesianResult:
    metric_name: str
    test_type: str               # "beta_binomial" or "normal_normal"
    control_mean: float
    treatment_mean: float
    absolute_lift: float
    relative_lift: float
    prob_treatment_better: float       # P(treat > ctrl)
    credible_low: float                # 2.5th percentile of (treat - ctrl)
    credible_high: float               # 97.5th percentile
    expected_loss_ship: float          # E[max(0, ctrl - treat)]
    expected_loss_no_ship: float       # E[max(0, treat - ctrl)]
    ship_recommended: bool
    control_n: int
    treatment_n: int


def _is_binary(s: pd.Series) -> bool:
    unique = pd.unique(s.dropna())
    return set(unique).issubset({0, 1, True, False})


def _sample_beta_posterior(
    successes: float, n: int, n_samples: int, rng: np.random.Generator
) -> np.ndarray:
    """Beta(1 + successes, 1 + failures) — conjugate posterior with Beta(1,1) prior."""
    return rng.beta(1 + successes, 1 + (n - successes), size=n_samples)


def _sample_normal_posterior(
    mean: float, var: float, n: int, n_samples: int, rng: np.random.Generator
) -> np.ndarray:
    """N(mean, var/n) — posterior of the mean under flat prior, plug-in variance."""
    se = np.sqrt(var / n) if var > 0 else 0.0
    if se == 0:
        return np.full(n_samples, mean)
    return rng.normal(mean, se, size=n_samples)


def _compute_loss_and_prob(
    ctrl_samples: np.ndarray, treat_samples: np.ndarray
) -> tuple[float, float, float, float, float]:
    """
    From paired posterior draws, compute:
      - P(treatment > control)
      - 95% credible interval on (treat - ctrl)
      - Expected loss if we ship the treatment
      - Expected loss if we don't ship
    """
    diff = treat_samples - ctrl_samples
    prob_better = float((diff > 0).mean())
    cred_low, cred_high = np.percentile(diff, [2.5, 97.5])

    loss_ship = float(np.maximum(0.0, -diff).mean())

    loss_no_ship = float(np.maximum(0.0, diff).mean())
    return prob_better, float(cred_low), float(cred_high), loss_ship, loss_no_ship


def run_bayesian_test(
    df: pd.DataFrame,
    column: str,
    metric_name: str,
    n_samples: int = N_SAMPLES,
    seed: int = SEED,
) -> BayesianResult:
    """Run a Bayesian comparison on `column` between variants."""
    ctrl = df.loc[df["variant"] == "control", column].dropna().to_numpy(dtype=float)
    treat = df.loc[df["variant"] == "treatment", column].dropna().to_numpy(dtype=float)

    rng = np.random.default_rng(seed)

    if _is_binary(pd.Series(np.concatenate([ctrl, treat]))):
        test_type = "beta_binomial"
        ctrl_samples = _sample_beta_posterior(ctrl.sum(), len(ctrl), n_samples, rng)
        treat_samples = _sample_beta_posterior(treat.sum(), len(treat), n_samples, rng)
    else:
        test_type = "normal_normal"
        ctrl_samples = _sample_normal_posterior(
            ctrl.mean(), ctrl.var(ddof=1), len(ctrl), n_samples, rng
        )
        treat_samples = _sample_normal_posterior(
            treat.mean(), treat.var(ddof=1), len(treat), n_samples, rng
        )

    prob, cred_low, cred_high, loss_ship, loss_no_ship = _compute_loss_and_prob(
        ctrl_samples, treat_samples
    )

    ctrl_mean = float(ctrl.mean())
    treat_mean = float(treat.mean())
    abs_lift = treat_mean - ctrl_mean
    rel_lift = abs_lift / ctrl_mean if ctrl_mean != 0 else 0.0

    # Ship if probability is high AND expected downside is small relative to baseline.
    loss_threshold = LOSS_SHIP_THRESHOLD_REL * abs(ctrl_mean) if ctrl_mean != 0 else float("inf")
    ship = (prob > PROB_SHIP_THRESHOLD) and (loss_ship < loss_threshold)

    return BayesianResult(
        metric_name=metric_name,
        test_type=test_type,
        control_mean=ctrl_mean,
        treatment_mean=treat_mean,
        absolute_lift=abs_lift,
        relative_lift=rel_lift,
        prob_treatment_better=prob,
        credible_low=cred_low,
        credible_high=cred_high,
        expected_loss_ship=loss_ship,
        expected_loss_no_ship=loss_no_ship,
        ship_recommended=bool(ship),
        control_n=len(ctrl),
        treatment_n=len(treat),
    )


PRIMARY_METRICS = [
    ("conversion_rate", "converted"),
    ("revenue_per_user", "revenue"),
]
SECONDARY_METRICS = [
    ("add_to_cart_per_user", "add_to_cart_events"),
    ("avg_pageviews", "pageviews"),
    ("avg_session_duration", "session_duration_sec"),
    ("avg_engagement_score", "engagement_score"),
]
GUARDRAIL_METRICS = [
    ("bounce_rate", "is_bounce_proxy"),
    ("session_depth", "pageviews"),
]


def run_all_bayesian(df: pd.DataFrame, analysis_unit: str = "user") -> dict:
    """
    Run Bayesian tests across primary, secondary, and guardrail tiers
    """
    if analysis_unit == "user":
        df = aggregate_to_user_level(df)
    elif analysis_unit != "session":
        raise ValueError(
            f"analysis_unit must be 'user' or 'session', got {analysis_unit!r}"
        )

    return {
        "analysis_unit": analysis_unit,
        "n_samples": N_SAMPLES,
        "primary": [_to_dict(run_bayesian_test(df, col, name)) for name, col in PRIMARY_METRICS],
        "secondary": [_to_dict(run_bayesian_test(df, col, name)) for name, col in SECONDARY_METRICS],
        "guardrail": [_to_dict(run_bayesian_test(df, col, name)) for name, col in GUARDRAIL_METRICS],
    }


def _to_dict(r: BayesianResult) -> dict:
    return {
        "metric_name": r.metric_name,
        "test_type": r.test_type,
        "control_mean": round(r.control_mean, 6),
        "treatment_mean": round(r.treatment_mean, 6),
        "absolute_lift": round(r.absolute_lift, 6),
        "relative_lift": round(r.relative_lift, 6),
        "prob_treatment_better": round(r.prob_treatment_better, 6),
        "credible_low": round(r.credible_low, 6),
        "credible_high": round(r.credible_high, 6),
        "expected_loss_ship": round(r.expected_loss_ship, 6),
        "expected_loss_no_ship": round(r.expected_loss_no_ship, 6),
        "ship_recommended": r.ship_recommended,
        "control_n": r.control_n,
        "treatment_n": r.treatment_n,
    }


if __name__ == "__main__":
    from clean import build_clean_sessions
    from assign_experiment import assign_users
    from simulate_treatment import simulate_treatment_effects

    df = build_clean_sessions()
    df = assign_users(df)
    df = simulate_treatment_effects(df)

    results = run_all_bayesian(df)

    for tier in ["primary", "secondary", "guardrail"]:
        print(f"\n{tier.upper()} METRICS  (analysis_unit={results['analysis_unit']})")
        for r in results[tier]:
            ship = "SHIP" if r["ship_recommended"] else "HOLD"
            print(
                f"[{ship}] {r['metric_name']:<24} "
                f"P(T>C)={r['prob_treatment_better']:.4f}  "
                f"lift={r['relative_lift']:+.2%}  "
                f"CrI=[{r['credible_low']:+.4f}, {r['credible_high']:+.4f}]  "
                f"L_ship={r['expected_loss_ship']:.5f}  "
                f"L_hold={r['expected_loss_no_ship']:.5f}"
            )

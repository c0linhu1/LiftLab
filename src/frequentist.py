"""
frequentist.py — Frequentist statistical testing for experiment metrics.


Explanation by claude:
Sessions are nested within users (one user has multiple sessions). Treating
sessions as IID violates the assumption of standard z-tests / t-tests,
inflates apparent significance, and is the #1 known pitfall of A/B testing
on session-level data.

This module addresses that by aggregating to USER level before testing — one
row per user, independence restored. The naive session-level path is still
exposed (analysis_unit="session") for didactic comparison, but should not
drive decisions.

Tests:
- Binary (0/1) at user level → two-proportion z-test
- Continuous at user level   → Welch's t-test (unequal variances)

For each metric: p-value, analytical 95% CI on absolute lift, bootstrap 95%
CI (10K iterations, chunked for memory), significance flag at α=0.05.

Also exposes srm_check() — sample-ratio-mismatch chi-squared on user counts.
SRM failure invalidates the experiment regardless of metric results.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy import stats


ALPHA = 0.05
BOOTSTRAP_ITERS = 10_000
BOOTSTRAP_CHUNK = 500   # iterations per chunk; bounds peak memory
SEED = 42


# How each session-level column collapses to a single user-level value.
# Independence is restored once every user contributes one row.
USER_AGG = {
    "converted": "max",                  # 1 if user converted in any session
    "transactions": "sum",
    "revenue": "sum",                    # total revenue per user
    "add_to_cart_events": "sum",
    "pageviews": "mean",                 # avg pageviews per session
    "event_count": "mean",
    "session_duration_sec": "mean",      # avg session duration
    "engagement_score": "mean",
    "is_bounce_proxy": "mean",           # bounce rate (sessions bounced / total)
    "device_type": "first",
    "country": "first",
    "variant": "first",                  # constant per user (user-level assignment)
}


@dataclass
class StatResult:
    metric_name: str
    test_type: str            # "proportions_z" or "welch_t"
    control_mean: float
    treatment_mean: float
    absolute_lift: float
    relative_lift: float
    p_value: float
    ci_low: float             # analytical 95% CI lower bound on absolute_lift
    ci_high: float
    boot_ci_low: float        # bootstrap 95% CI lower bound
    boot_ci_high: float
    significant: bool         # p_value < ALPHA
    control_n: int
    treatment_n: int


def aggregate_to_user_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse session-level data to one row per user using USER_AGG rules.
    Restores independence for downstream tests.
    """
    cols = {k: v for k, v in USER_AGG.items() if k in df.columns}
    return df.groupby("user_pseudo_id", as_index=False).agg(cols)


def srm_check(df: pd.DataFrame, expected_treatment_share: float = 0.5) -> dict:
    """
    Sample-Ratio-Mismatch test on user-level variant counts.

    A failing SRM means the observed split deviates from the configured one
    by more than chance — almost always a bug in assignment, logging, or
    filtering. When SRM fails, no metric result from this experiment is
    trustworthy.
    """
    user_variant = df.groupby("user_pseudo_id")["variant"].first()
    n = int(len(user_variant))
    obs_t = int((user_variant == "treatment").sum())
    obs_c = n - obs_t
    exp_t = n * expected_treatment_share
    exp_c = n * (1 - expected_treatment_share)
    chi2 = (obs_t - exp_t) ** 2 / exp_t + (obs_c - exp_c) ** 2 / exp_c
    p_value = float(1 - stats.chi2.cdf(chi2, df=1))

    return {
        "n_users": n,
        "n_treatment": obs_t,
        "n_control": obs_c,
        "expected_treatment_share": expected_treatment_share,
        "actual_treatment_share": round(obs_t / n, 6),
        "chi2": round(float(chi2), 6),
        "p_value": round(p_value, 6),
        "passed": p_value > 0.001,   # standard SRM threshold
    }


def _is_binary(s: pd.Series) -> bool:
    """A column is binary if it only contains 0/1 (or True/False)."""
    unique = pd.unique(s.dropna())
    return set(unique).issubset({0, 1, True, False})


def _proportions_z_test(
    ctrl: np.ndarray, treat: np.ndarray
) -> tuple[float, float, float]:
    """
    Two-proportion z-test.

    Pooled SE under H0 for the p-value, unpooled SE for the CI.
    Returns (p_value, ci_low, ci_high) for the absolute lift.
    """
    n1, n2 = len(treat), len(ctrl)
    x1, x2 = treat.sum(), ctrl.sum()
    p1, p2 = x1 / n1, x2 / n2
    diff = p1 - p2

    p_pool = (x1 + x2) / (n1 + n2)
    se_pooled = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se_pooled == 0:
        return 1.0, diff, diff

    z = diff / se_pooled
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    se_unpooled = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    z_crit = stats.norm.ppf(1 - ALPHA / 2)
    return p_value, diff - z_crit * se_unpooled, diff + z_crit * se_unpooled


def _welch_t_test(
    ctrl: np.ndarray, treat: np.ndarray
) -> tuple[float, float, float]:
    """
    Welch's t-test for difference of means (no equal-variance assumption).
    Returns (p_value, ci_low, ci_high) for the absolute lift.
    """
    n1, n2 = len(treat), len(ctrl)
    m1, m2 = treat.mean(), ctrl.mean()
    v1, v2 = treat.var(ddof=1), ctrl.var(ddof=1)
    diff = m1 - m2

    se = np.sqrt(v1 / n1 + v2 / n2)
    if se == 0:
        return 1.0, diff, diff

    t = diff / se
    df = (v1 / n1 + v2 / n2) ** 2 / (
        (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
    )
    p_value = 2 * (1 - stats.t.cdf(abs(t), df))
    t_crit = stats.t.ppf(1 - ALPHA / 2, df)
    return p_value, diff - t_crit * se, diff + t_crit * se


def _bootstrap_ci(
    ctrl: np.ndarray,
    treat: np.ndarray,
    n_iters: int = BOOTSTRAP_ITERS,
    chunk: int = BOOTSTRAP_CHUNK,
    seed: int = SEED,
) -> tuple[float, float]:
    """
    Empirical 95% CI on (treat_mean - ctrl_mean), chunked so peak memory
    stays bounded regardless of input size.
    """
    rng = np.random.default_rng(seed)
    n1, n2 = len(treat), len(ctrl)
    diffs = np.empty(n_iters, dtype=np.float64)

    for start in range(0, n_iters, chunk):
        k = min(chunk, n_iters - start)
        ti = rng.integers(0, n1, size=(k, n1))
        ci = rng.integers(0, n2, size=(k, n2))
        diffs[start:start + k] = treat[ti].mean(axis=1) - ctrl[ci].mean(axis=1)

    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def run_test(df: pd.DataFrame, column: str, metric_name: str) -> StatResult:
    """Run the appropriate test on `column` between variants."""
    ctrl = df.loc[df["variant"] == "control", column].dropna().to_numpy()
    treat = df.loc[df["variant"] == "treatment", column].dropna().to_numpy()

    ctrl_mean = float(ctrl.mean())
    treat_mean = float(treat.mean())
    abs_lift = treat_mean - ctrl_mean
    rel_lift = abs_lift / ctrl_mean if ctrl_mean != 0 else 0.0

    if _is_binary(pd.Series(np.concatenate([ctrl, treat]))):
        test_type = "proportions_z"
        p, lo, hi = _proportions_z_test(ctrl, treat)
    else:
        test_type = "welch_t"
        p, lo, hi = _welch_t_test(ctrl, treat)

    boot_lo, boot_hi = _bootstrap_ci(ctrl, treat)

    return StatResult(
        metric_name=metric_name,
        test_type=test_type,
        control_mean=ctrl_mean,
        treatment_mean=treat_mean,
        absolute_lift=abs_lift,
        relative_lift=rel_lift,
        p_value=float(p),
        ci_low=float(lo),
        ci_high=float(hi),
        boot_ci_low=boot_lo,
        boot_ci_high=boot_hi,
        significant=bool(p < ALPHA),
        control_n=len(ctrl),
        treatment_n=len(treat),
    )


# At user level: converted = max (ever-convert), revenue = sum (total),
# pageviews = mean (avg per session), engagement = mean, etc.
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


def run_all_tests(df: pd.DataFrame, analysis_unit: str = "user") -> dict:
    """
    Run frequentist tests across primary, secondary, and guardrail tiers.

    analysis_unit:
      "user"     — aggregate to user level first (default, correct).
      "session"  — naive session-level analysis. Provided for comparison;
                   will inflate significance due to within-user correlation.
    """
    if analysis_unit == "user":
        df = aggregate_to_user_level(df)
    elif analysis_unit != "session":
        raise ValueError(
            f"analysis_unit must be 'user' or 'session', got {analysis_unit!r}"
        )

    return {
        "analysis_unit": analysis_unit,
        "primary": [_to_dict(run_test(df, col, name)) for name, col in PRIMARY_METRICS],
        "secondary": [_to_dict(run_test(df, col, name)) for name, col in SECONDARY_METRICS],
        "guardrail": [_to_dict(run_test(df, col, name)) for name, col in GUARDRAIL_METRICS],
    }


def _to_dict(r: StatResult) -> dict:
    return {
        "metric_name": r.metric_name,
        "test_type": r.test_type,
        "control_mean": round(r.control_mean, 6),
        "treatment_mean": round(r.treatment_mean, 6),
        "absolute_lift": round(r.absolute_lift, 6),
        "relative_lift": round(r.relative_lift, 6),
        "p_value": round(r.p_value, 6),
        "ci_low": round(r.ci_low, 6),
        "ci_high": round(r.ci_high, 6),
        "boot_ci_low": round(r.boot_ci_low, 6),
        "boot_ci_high": round(r.boot_ci_high, 6),
        "significant": r.significant,
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

    print("SAMPLE RATIO MISMATCH CHECK")
    srm = srm_check(df)
    print(
        f"  users={srm['n_users']:,}  "
        f"treat={srm['n_treatment']:,}  "
        f"ctrl={srm['n_control']:,}  "
        f"actual={srm['actual_treatment_share']:.4f}  "
        f"chi2={srm['chi2']:.4f}  "
        f"p={srm['p_value']:.4f}  "
        f"{'PASS' if srm['passed'] else 'FAIL'}"
    )

    results = run_all_tests(df)
    for tier in ["primary", "secondary", "guardrail"]:
        print(f"\n=== {tier.upper()} METRICS  (analysis_unit={results['analysis_unit']}) ===")
        for r in results[tier]:
            sig = "*" if r["significant"] else " "
            print(
                f"  {sig} {r['metric_name']:<24} "
                f"lift={r['relative_lift']:+.2%}  "
                f"p={r['p_value']:.4f}  "
                f"CI=[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]  "
                f"boot=[{r['boot_ci_low']:+.4f}, {r['boot_ci_high']:+.4f}]"
            )

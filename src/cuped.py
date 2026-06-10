"""
cuped.py — CUPED variance reduction for experiment metrics.



Explanation by claude:
CUPED (Controlled-experiment Using Pre-Experiment Data, Microsoft 2013)
adjusts each user's outcome Y using their pre-experiment behavior X:

    Y_adj = Y - (X - X̄) @ θ,   θ fit by OLS on combined data

This subtracts predictable, pre-existing variation from Y without
biasing the treatment-control difference (since assignment is random,
E[X | variant] is the same for both variants in expectation).

Net effect: smaller Var(Y_adj) → smaller SE → narrower CI / smaller p-value,
with no change to the point estimate of the lift. Variance reduction is
bounded by the R² of the Y~X regression — strong pre-experiment predictors
are what make CUPED work.

θ is fit on COMBINED data (control + treatment together) to avoid leaking
the treatment signal into the adjustment.

Covariates used (built leakage-safely in clean.py):
- prior_sessions       — # sessions before this one
- prior_revenue        — cumulative revenue before this session
- prior_avg_engagement — expanding mean of engagement before this session

For each user we take the MIN of each prior_* across their experiment-
window sessions, which corresponds to their state at experiment entry
(prior_* is cumulative up to but excluding the current session).
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass

from frequentist import ALPHA, USER_AGG, _welch_t_test


COVARIATES = ["prior_sessions", "prior_revenue", "prior_avg_engagement"]


METRICS_FOR_CUPED = [
    ("conversion_rate", "converted"),
    ("revenue_per_user", "revenue"),
    ("add_to_cart_per_user", "add_to_cart_events"),
    ("avg_pageviews", "pageviews"),
    ("avg_session_duration", "session_duration_sec"),
    ("avg_engagement_score", "engagement_score"),
    ("bounce_rate", "is_bounce_proxy"),
]


@dataclass
class CupedResult:
    metric_name: str
    raw_lift: float
    raw_p_value: float
    raw_ci_low: float
    raw_ci_high: float
    adj_lift: float
    adj_p_value: float
    adj_ci_low: float
    adj_ci_high: float
    variance_reduction: float    # 1 - Var(Y_adj)/Var(Y), higher = better
    ci_width_improvement: float  # 1 - width_adj/width_raw, higher = better
    theta: list                  # OLS coefficients, one per covariate
    control_n: int
    treatment_n: int
    significant_raw: bool
    significant_adj: bool


def aggregate_with_covariates(df: pd.DataFrame) -> pd.DataFrame:
    """
    User-level aggregation including prior_* covariates.
    """
    outcome_agg = {k: v for k, v in USER_AGG.items() if k in df.columns}
    cov_agg = {c: "min" for c in COVARIATES if c in df.columns}
    return df.groupby("user_pseudo_id", as_index=False).agg(
        {**outcome_agg, **cov_agg}
    )


def _cuped_theta(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    OLS coefficients regressing Y on centered X.

    Uses lstsq with rcond=None so degenerate covariates (zero variance,
    collinear) don't blow up — affected coefficients fall to ~0 and the
    adjustment becomes a no-op for that covariate.
    """
    X_centered = X - X.mean(axis=0)
    y_centered = y - y.mean()
    theta, *_ = np.linalg.lstsq(X_centered, y_centered, rcond=None)
    return theta


def cuped_adjust(
    y: np.ndarray, X: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Y_adj = Y - (X - X̄) @ θ, θ fit on the COMBINED data passed in.
    Returns (y_adjusted, theta).
    """
    theta = _cuped_theta(y, X)
    X_centered = X - X.mean(axis=0)
    y_adj = y - X_centered @ theta
    return y_adj, theta


def run_cuped_test(
    df_user: pd.DataFrame,
    column: str,
    metric_name: str,
    covariates: list[str] = COVARIATES,
) -> CupedResult:
    """
    Run raw vs CUPED-adjusted Welch's t-test on user-level data.
    """
    y = df_user[column].to_numpy(dtype=float)
    X = df_user[covariates].to_numpy(dtype=float)
    variant = df_user["variant"].to_numpy()

    y_adj, theta = cuped_adjust(y, X)

    ctrl_mask = variant == "control"
    treat_mask = variant == "treatment"

    y_ctrl, y_treat = y[ctrl_mask], y[treat_mask]
    yadj_ctrl, yadj_treat = y_adj[ctrl_mask], y_adj[treat_mask]

    raw_p, raw_lo, raw_hi = _welch_t_test(y_ctrl, y_treat)
    raw_lift = float(y_treat.mean() - y_ctrl.mean())

    adj_p, adj_lo, adj_hi = _welch_t_test(yadj_ctrl, yadj_treat)
    adj_lift = float(yadj_treat.mean() - yadj_ctrl.mean())

    var_raw = float(np.var(y, ddof=1))
    var_adj = float(np.var(y_adj, ddof=1))
    var_reduction = 1 - var_adj / var_raw if var_raw > 0 else 0.0

    width_raw = raw_hi - raw_lo
    width_adj = adj_hi - adj_lo
    ci_improvement = 1 - width_adj / width_raw if width_raw > 0 else 0.0

    return CupedResult(
        metric_name=metric_name,
        raw_lift=raw_lift,
        raw_p_value=float(raw_p),
        raw_ci_low=float(raw_lo),
        raw_ci_high=float(raw_hi),
        adj_lift=adj_lift,
        adj_p_value=float(adj_p),
        adj_ci_low=float(adj_lo),
        adj_ci_high=float(adj_hi),
        variance_reduction=float(var_reduction),
        ci_width_improvement=float(ci_improvement),
        theta=[float(t) for t in theta],
        control_n=int(ctrl_mask.sum()),
        treatment_n=int(treat_mask.sum()),
        significant_raw=bool(raw_p < ALPHA),
        significant_adj=bool(adj_p < ALPHA),
    )


def run_all_cuped(
    df: pd.DataFrame, covariates: list[str] = COVARIATES
) -> dict:
    """Aggregate to user level (with covariates), run CUPED on every metric."""
    df_user = aggregate_with_covariates(df)
    available = [c for c in covariates if c in df_user.columns]
    if not available:
        raise ValueError(f"None of {covariates} present in df.")

    return {
        "covariates": available,
        "n_users": int(len(df_user)),
        "results": [
            _to_dict(run_cuped_test(df_user, col, name, available))
            for name, col in METRICS_FOR_CUPED
            if col in df_user.columns
        ],
    }


def _to_dict(r: CupedResult) -> dict:
    return {
        "metric_name": r.metric_name,
        "raw_lift": round(r.raw_lift, 6),
        "raw_p_value": round(r.raw_p_value, 6),
        "raw_ci_low": round(r.raw_ci_low, 6),
        "raw_ci_high": round(r.raw_ci_high, 6),
        "adj_lift": round(r.adj_lift, 6),
        "adj_p_value": round(r.adj_p_value, 6),
        "adj_ci_low": round(r.adj_ci_low, 6),
        "adj_ci_high": round(r.adj_ci_high, 6),
        "variance_reduction": round(r.variance_reduction, 6),
        "ci_width_improvement": round(r.ci_width_improvement, 6),
        "theta": [round(t, 6) for t in r.theta],
        "control_n": r.control_n,
        "treatment_n": r.treatment_n,
        "significant_raw": r.significant_raw,
        "significant_adj": r.significant_adj,
    }


if __name__ == "__main__":
    from clean import build_clean_sessions
    from assign_experiment import assign_users
    from simulate_treatment import simulate_treatment_effects

    df = build_clean_sessions()
    df = assign_users(df)
    df = simulate_treatment_effects(df)

    output = run_all_cuped(df)
    print(
        f"CUPED RESULTS (n_users={output['n_users']:,}, "
        f"covariates: {', '.join(output['covariates'])})\n"
    )
    header = (
        f"  {'metric':<24} {'raw lift':>10} {'raw p':>8}  "
        f"{'adj lift':>10} {'adj p':>8}  {'var red':>8} {'CI shrink':>10}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in output["results"]:
        sig_raw = "*" if r["significant_raw"] else " "
        sig_adj = "*" if r["significant_adj"] else " "
        print(
            f"  {r['metric_name']:<24} "
            f"{r['raw_lift']:+10.4f}{sig_raw}{r['raw_p_value']:7.4f}  "
            f"{r['adj_lift']:+10.4f}{sig_adj}{r['adj_p_value']:7.4f}  "
            f"{r['variance_reduction']:>7.1%} {r['ci_width_improvement']:>9.1%}"
        )

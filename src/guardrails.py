"""
guardrails.py — Threshold-based guardrail evaluation.

Independent of statistical significance: a guardrail can be "significantly
worse" by t-test but still within the acceptable threshold (small effect,
big n), and vice versa. The question here is policy, not inference — did
the relative change exceed the limit?

Thresholds (relative change in the policy-bad direction):
- bounce_rate:        FAIL if treatment is >+5% higher than control
- session_depth:      FAIL if treatment is >−10% lower than control
- revenue_volatility: FAIL if treatment std is >+15% higher than control

WARNING thresholds set at half the FAIL value, so we get early signal
before a policy limit is breached.

Direction-aware: each guardrail has a `direction` flag so the logic
correctly treats a bounce_rate DROP as PASS and a SPIKE as FAIL. This
fixes the interpretation quirk in bayesian.py where P(T>C)=0 on bounces
read as "HOLD" when it should read as "good news."

Aggregation:
- bounce_rate, session_depth — evaluated at USER level (consistent with
  rest of platform).
- revenue_volatility — evaluated at SESSION level because purchaser
  status is per-session; user-level aggregation would conflate users
  with multiple purchases.
"""

from dataclasses import dataclass
import pandas as pd

from frequentist import aggregate_to_user_level


@dataclass
class GuardrailSpec:
    name: str
    column: str
    direction: str            # "lower_is_better" or "higher_is_better"
    fail_threshold: float     # relative regression magnitude that triggers FAIL


@dataclass
class GuardrailResult:
    name: str
    column: str
    control_value: float
    treatment_value: float
    absolute_diff: float
    relative_diff: float
    direction: str
    fail_threshold: float
    warning_threshold: float
    status: str               # "pass", "warning", "fail"


GUARDRAILS = [
    GuardrailSpec("bounce_rate", "is_bounce_proxy", "lower_is_better", 0.05),
    GuardrailSpec("session_depth", "pageviews", "higher_is_better", 0.10),
]
REVENUE_VOLATILITY_THRESHOLD = 0.15


def _status_from_relative_diff(
    rel_diff: float, direction: str, fail: float
) -> str:
    """
    rel_diff = (treat - ctrl) / ctrl. Convert to a "regression magnitude"
    (positive = worse, negative = better) using direction, then classify.
    """
    warning = fail / 2
    if direction == "lower_is_better":
        regression = rel_diff       # going up is bad
    elif direction == "higher_is_better":
        regression = -rel_diff      # going down is bad
    else:
        raise ValueError(f"Unknown direction: {direction!r}")

    if regression >= fail:
        return "fail"
    if regression >= warning:
        return "warning"
    return "pass"


def evaluate_guardrail(
    df_user: pd.DataFrame, spec: GuardrailSpec
) -> GuardrailResult:
    """Standard mean-based guardrail check at user level."""
    ctrl = df_user.loc[df_user["variant"] == "control", spec.column]
    treat = df_user.loc[df_user["variant"] == "treatment", spec.column]
    ctrl_mean = float(ctrl.mean())
    treat_mean = float(treat.mean())
    abs_diff = treat_mean - ctrl_mean
    rel_diff = abs_diff / ctrl_mean if ctrl_mean != 0 else 0.0
    return GuardrailResult(
        name=spec.name,
        column=spec.column,
        control_value=ctrl_mean,
        treatment_value=treat_mean,
        absolute_diff=abs_diff,
        relative_diff=rel_diff,
        direction=spec.direction,
        fail_threshold=spec.fail_threshold,
        warning_threshold=spec.fail_threshold / 2,
        status=_status_from_relative_diff(rel_diff, spec.direction, spec.fail_threshold),
    )


def evaluate_revenue_volatility(df: pd.DataFrame) -> GuardrailResult:
    """
    Volatility = std of revenue among purchasers (revenue > 0), per variant.
    Computed at SESSION level — purchaser status is per-session, and a
    user-level aggregation would smear sessions across multiple purchases.
    Higher std means more erratic spend, which is the policy-bad direction.
    """
    purchasers = df[df["revenue"] > 0]
    ctrl_std = float(
        purchasers.loc[purchasers["variant"] == "control", "revenue"].std(ddof=1)
    )
    treat_std = float(
        purchasers.loc[purchasers["variant"] == "treatment", "revenue"].std(ddof=1)
    )
    abs_diff = treat_std - ctrl_std
    rel_diff = abs_diff / ctrl_std if ctrl_std > 0 else 0.0
    return GuardrailResult(
        name="revenue_volatility",
        column="revenue",
        control_value=ctrl_std,
        treatment_value=treat_std,
        absolute_diff=abs_diff,
        relative_diff=rel_diff,
        direction="lower_is_better",
        fail_threshold=REVENUE_VOLATILITY_THRESHOLD,
        warning_threshold=REVENUE_VOLATILITY_THRESHOLD / 2,
        status=_status_from_relative_diff(
            rel_diff, "lower_is_better", REVENUE_VOLATILITY_THRESHOLD
        ),
    )


def run_all_guardrails(df: pd.DataFrame) -> dict:
    """
    Evaluate all guardrails and compute overall status.
    overall_status is the worst individual status — any FAIL blocks rollout.
    """
    df_user = aggregate_to_user_level(df)
    results = [
        _to_dict(evaluate_guardrail(df_user, spec))
        for spec in GUARDRAILS
        if spec.column in df_user.columns
    ]
    results.append(_to_dict(evaluate_revenue_volatility(df)))

    if any(r["status"] == "fail" for r in results):
        overall = "fail"
    elif any(r["status"] == "warning" for r in results):
        overall = "warning"
    else:
        overall = "pass"

    return {"overall_status": overall, "results": results}


def _to_dict(r: GuardrailResult) -> dict:
    return {
        "name": r.name,
        "column": r.column,
        "control_value": round(r.control_value, 6),
        "treatment_value": round(r.treatment_value, 6),
        "absolute_diff": round(r.absolute_diff, 6),
        "relative_diff": round(r.relative_diff, 6),
        "direction": r.direction,
        "fail_threshold": r.fail_threshold,
        "warning_threshold": r.warning_threshold,
        "status": r.status,
    }


if __name__ == "__main__":
    from clean import build_clean_sessions
    from assign_experiment import assign_users
    from simulate_treatment import simulate_treatment_effects

    df = build_clean_sessions()
    df = assign_users(df)
    df = simulate_treatment_effects(df)

    output = run_all_guardrails(df)

    print(f"\nGUARDRAIL CHECK — OVERALL: {output['overall_status'].upper()}\n")
    print(f"  {'guardrail':<22} {'ctrl':>10} {'treat':>10} {'rel diff':>10} {'limit':>8}  status")
    for r in output["results"]:
        sign = "+" if r["relative_diff"] >= 0 else ""
        good = "↑" if r["direction"] == "higher_is_better" else "↓"
        print(
            f"  {r['name']:<22} "
            f"{r['control_value']:>10.4f} "
            f"{r['treatment_value']:>10.4f} "
            f"{sign}{r['relative_diff']:>9.2%} "
            f"{r['fail_threshold']:>7.0%}  "
            f"{r['status'].upper()}  ({good} good)"
        )

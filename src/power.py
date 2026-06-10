"""
power.py — Pre-experiment power analysis and minimum detectable effect (MDE).


Explanation by claude again
Two questions every DS lead asks before greenlighting an experiment:
1. "How many users per arm do I need to detect a Δ% lift with 80% power?"
2. "Given the n I'll realistically get, what's the smallest effect I could
   detect — am I even powered to find what I'm looking for?"

This module answers both, for binary outcomes (two-proportion normal-
approximation test) and continuous outcomes (two-mean t-approximation).
It also exposes `analyze_experiment_power` which inspects a USER-LEVEL
DataFrame and reports the MDE for every primary metric given the data
that's already there — useful for diagnosing "we ran an experiment and
nothing was significant; was it underpowered?"

Formulas (standard two-sided test, equal allocation):

  Binary:
    n_per_arm = ((z_{α/2} + z_β) · √(2·p̄·(1−p̄))) / Δ )²
    MDE       = (z_{α/2} + z_β) · √(2·p̄·(1−p̄) / n)

  Continuous:
    n_per_arm = 2 · ((z_{α/2} + z_β) · σ / Δ)²
    MDE       = (z_{α/2} + z_β) · √(2σ² / n)

These use the normal approximation; for tiny samples you'd want exact /
simulation-based power, but at A/B-test scale (n in the thousands+) the
approximation is sharp.
"""

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
from scipy import stats

from frequentist import aggregate_to_user_level, PRIMARY_METRICS


ALPHA = 0.05
DEFAULT_POWER = 0.80


@dataclass
class PowerResult:
    metric_name: str
    test_type: str            # "binary" or "continuous"
    baseline: float           # baseline rate or mean
    n_per_arm_observed: int   # smaller of the two arms in the input data
    mde_absolute: float       # smallest detectable absolute lift
    mde_relative: float       # ditto as a fraction of baseline
    alpha: float
    power: float


def _z(alpha: float, power: float) -> tuple[float, float]:
    """Critical values for a two-sided test at level alpha and target power."""
    return stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)


def sample_size_binary(
    p_baseline: float,
    mde_relative: float,
    alpha: float = ALPHA,
    power: float = DEFAULT_POWER,
) -> int:
    """
    Users per arm to detect a `mde_relative` lift on `p_baseline`.

    p_baseline      — control conversion rate (e.g., 0.05 for 5%)
    mde_relative    — target lift as a fraction of baseline (e.g., 0.10 for +10%)
    """
    z_a, z_b = _z(alpha, power)
    p1 = p_baseline
    p2 = p_baseline * (1 + mde_relative)
    p_bar = (p1 + p2) / 2
    delta = abs(p2 - p1)
    if delta == 0:
        return math.inf
    n = ((z_a + z_b) * math.sqrt(2 * p_bar * (1 - p_bar)) / delta) ** 2
    return math.ceil(n)


def sample_size_continuous(
    mean_baseline: float,
    std: float,
    mde_relative: float,
    alpha: float = ALPHA,
    power: float = DEFAULT_POWER,
) -> int:
    """Users per arm to detect `mde_relative` lift on a continuous metric."""
    z_a, z_b = _z(alpha, power)
    delta = abs(mean_baseline * mde_relative)
    if delta == 0 or std == 0:
        return math.inf
    n = 2 * ((z_a + z_b) * std / delta) ** 2
    return math.ceil(n)


def mde_binary(
    n_per_arm: int,
    p_baseline: float,
    alpha: float = ALPHA,
    power: float = DEFAULT_POWER,
) -> float:
    """Smallest absolute lift detectable at this n, baseline rate, α and power."""
    z_a, z_b = _z(alpha, power)
    p_bar = p_baseline   # under H0, p1=p2=p_baseline so p_bar=p_baseline
    return (z_a + z_b) * math.sqrt(2 * p_bar * (1 - p_bar) / n_per_arm)


def mde_continuous(
    n_per_arm: int,
    std: float,
    alpha: float = ALPHA,
    power: float = DEFAULT_POWER,
) -> float:
    """Smallest absolute lift detectable on a continuous metric at this n."""
    z_a, z_b = _z(alpha, power)
    return (z_a + z_b) * math.sqrt(2 * std ** 2 / n_per_arm)


def _is_binary(s: pd.Series) -> bool:
    """Matches frequentist._is_binary, kept local to avoid cross-import."""
    unique = pd.unique(s.dropna())
    return set(unique).issubset({0, 1, True, False})


def analyze_experiment_power(
    df: pd.DataFrame,
    alpha: float = ALPHA,
    power: float = DEFAULT_POWER,
    metrics: list = PRIMARY_METRICS,
) -> list:
    """
    For each primary metric, report the MDE at the n we actually have.

    Aggregates to user level first (consistent with frequentist.py and the
    rest of the platform). Uses the CONTROL arm's mean/std as the baseline.
    """
    df_user = aggregate_to_user_level(df)
    n_ctrl = int((df_user["variant"] == "control").sum())
    n_treat = int((df_user["variant"] == "treatment").sum())
    n_per_arm = min(n_ctrl, n_treat)

    results = []
    for metric_name, column in metrics:
        if column not in df_user.columns:
            continue
        ctrl = df_user.loc[df_user["variant"] == "control", column].dropna().to_numpy(dtype=float)
        baseline = float(ctrl.mean())

        if _is_binary(pd.Series(ctrl)):
            test_type = "binary"
            mde_abs = mde_binary(n_per_arm, baseline, alpha=alpha, power=power)
        else:
            test_type = "continuous"
            std = float(ctrl.std(ddof=1))
            mde_abs = mde_continuous(n_per_arm, std, alpha=alpha, power=power)

        mde_rel = mde_abs / baseline if baseline != 0 else float("inf")

        results.append(PowerResult(
            metric_name=metric_name,
            test_type=test_type,
            baseline=baseline,
            n_per_arm_observed=n_per_arm,
            mde_absolute=mde_abs,
            mde_relative=mde_rel,
            alpha=alpha,
            power=power,
        ))

    return results


def required_sample_for_experiment(
    df: pd.DataFrame,
    target_relative_lifts: dict,
    alpha: float = ALPHA,
    power: float = DEFAULT_POWER,
    metrics: list = PRIMARY_METRICS,
) -> list:
    """
    For each primary metric, report users-per-arm required to detect the
    target relative lift given the metric's natural variance (estimated
    from the control arm). Use this BEFORE running an experiment.

    target_relative_lifts: dict mapping metric_name -> desired lift fraction
    """
    df_user = aggregate_to_user_level(df)
    results = []

    for metric_name, column in metrics:
        if column not in df_user.columns:
            continue
        target = target_relative_lifts.get(metric_name)
        if target is None:
            continue
        ctrl = df_user.loc[df_user["variant"] == "control", column].dropna().to_numpy(dtype=float)
        baseline = float(ctrl.mean())

        if _is_binary(pd.Series(ctrl)):
            n_needed = sample_size_binary(baseline, target, alpha=alpha, power=power)
            test_type = "binary"
        else:
            std = float(ctrl.std(ddof=1))
            n_needed = sample_size_continuous(baseline, std, target, alpha=alpha, power=power)
            test_type = "continuous"

        results.append({
            "metric_name": metric_name,
            "test_type": test_type,
            "baseline": round(baseline, 6),
            "target_relative_lift": target,
            "n_per_arm_required": n_needed,
            "alpha": alpha,
            "power": power,
        })

    return results


def _to_dict(r: PowerResult) -> dict:
    return {
        "metric_name": r.metric_name,
        "test_type": r.test_type,
        "baseline": round(r.baseline, 6),
        "n_per_arm_observed": r.n_per_arm_observed,
        "mde_absolute": round(r.mde_absolute, 6),
        "mde_relative": round(r.mde_relative, 6),
        "alpha": r.alpha,
        "power": r.power,
    }


if __name__ == "__main__":
    from clean import build_clean_sessions
    from assign_experiment import assign_users
    from simulate_treatment import simulate_treatment_effects

    df = build_clean_sessions()
    df = assign_users(df)
    df = simulate_treatment_effects(df)

    print("\n=== POST-HOC POWER (what's detectable given the n we have) ===\n")
    results = analyze_experiment_power(df)
    print(f"  {'metric':<22} {'baseline':>10} {'n/arm':>10} {'MDE abs':>12} {'MDE rel':>10}")
    print("  " + "-" * 70)
    for r in results:
        print(
            f"  {r.metric_name:<22} "
            f"{r.baseline:>10.4f} "
            f"{r.n_per_arm_observed:>10,} "
            f"{r.mde_absolute:>12.6f} "
            f"{r.mde_relative:>9.2%}"
        )

    print("\n=== PRE-EXPERIMENT SAMPLE-SIZE TARGETS ===\n")
    print("  (Required n per arm to detect a +5% relative lift)\n")
    required = required_sample_for_experiment(
        df,
        target_relative_lifts={"conversion_rate": 0.05, "revenue_per_user": 0.05},
    )
    print(f"  {'metric':<22} {'baseline':>10} {'target lift':>12} {'n/arm needed':>16}")
    print("  " + "-" * 70)
    for r in required:
        print(
            f"  {r['metric_name']:<22} "
            f"{r['baseline']:>10.4f} "
            f"{r['target_relative_lift']:>11.0%} "
            f"{r['n_per_arm_required']:>16,}"
        )

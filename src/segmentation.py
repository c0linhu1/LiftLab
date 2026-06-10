"""
segmentation.py — Heterogeneous treatment effect analysis.

Runs the frequentist test from frequentist.py within each segment of the
data — device type, top-N countries, traffic source, traffic medium —
to surface treatment effects that vary by user group.

A treatment that's strongly positive in one segment and flat (or negative)
in another is a heterogeneous effect. Pooling them in the topline analysis
averages those signals together and hides the truth. Segmentation is what
catches "ships well to mobile, hurts desktop" stories.

Sample-size floor per variant per segment is enforced (default 100 users)
because small segments produce noisy p-values that flap between significant
and not. Segments below the threshold are emitted with `underpowered=True`
and no metric results, so downstream consumers can flag them rather than
silently include flaky numbers.

Tests are run on PRIMARY metrics only (conversion_rate, revenue_per_user)
— segmentation is about decision-driving metrics, not diagnostic ones.
"""

import pandas as pd
from frequentist import (
    aggregate_to_user_level,
    run_test,
    PRIMARY_METRICS,
    _to_dict as _stat_to_dict,
)


MIN_USERS_PER_VARIANT = 100
TOP_N_COUNTRIES = 10

SEGMENT_COLUMNS = ["device_type", "country", "traffic_source", "traffic_medium"]


def _top_n_segments(df: pd.DataFrame, col: str, n: int) -> list:
    """Return the top-N values of `col` by unique user count."""
    return (
        df.groupby(col)["user_pseudo_id"]
        .nunique()
        .sort_values(ascending=False)
        .head(n)
        .index
        .tolist()
    )


def run_segment_tests(
    df: pd.DataFrame,
    segment_col: str,
    segment_values: list,
    metrics: list = PRIMARY_METRICS,
    min_users: int = MIN_USERS_PER_VARIANT,
) -> list:
    """
    Run primary-metric tests within each value of `segment_col`.
    """
    results = []

    for value in segment_values:
        sub = df[df[segment_col] == value]
        user_df = aggregate_to_user_level(sub)

        n_ctrl = int((user_df["variant"] == "control").sum())
        n_treat = int((user_df["variant"] == "treatment").sum())
        underpowered = (n_ctrl < min_users) or (n_treat < min_users)

        segment_result = {
            "segment_col": segment_col,
            "segment_value": str(value),
            "n_users_control": n_ctrl,
            "n_users_treatment": n_treat,
            "underpowered": underpowered,
            "metrics": [],
        }

        if not underpowered:
            for metric_name, column in metrics:
                if column not in user_df.columns:
                    continue
                stat = run_test(user_df, column, metric_name)
                segment_result["metrics"].append(_stat_to_dict(stat))

        results.append(segment_result)

    return results


def run_all_segmentation(
    df: pd.DataFrame,
    segment_cols: list = SEGMENT_COLUMNS,
    top_n_countries: int = TOP_N_COUNTRIES,
    min_users: int = MIN_USERS_PER_VARIANT,
    metrics: list = PRIMARY_METRICS,
) -> dict:
    """Run segmentation across every standard segment column."""
    output = {
        "min_users_per_variant": min_users,
        "metrics_tested": [m[0] for m in metrics],
        "segments": {},
    }

    for col in segment_cols:
        if col not in df.columns:
            continue
        if col == "country":
            values = _top_n_segments(df, col, top_n_countries)
        else:
            values = sorted(v for v in df[col].dropna().unique())

        output["segments"][col] = run_segment_tests(
            df, col, values, metrics=metrics, min_users=min_users
        )

    return output


if __name__ == "__main__":
    from clean import build_clean_sessions
    from assign_experiment import assign_users
    from simulate_treatment import simulate_treatment_effects

    df = build_clean_sessions()
    df = assign_users(df)
    df = simulate_treatment_effects(df)

    output = run_all_segmentation(df)
    metrics_tested = ", ".join(output["metrics_tested"])
    print(
        f"Segmentation (min {output['min_users_per_variant']} users/variant, "
        f"metrics: {metrics_tested})"
    )

    for seg_col, results in output["segments"].items():
        print(f"\n=BY {seg_col.upper()}")
        for seg in results:
            tag = "  [UNDERPOWERED]" if seg["underpowered"] else ""
            header = (
                f"  {seg['segment_value']:<30} "
                f"n_ctrl={seg['n_users_control']:>6,}  "
                f"n_treat={seg['n_users_treatment']:>6,}{tag}"
            )
            print(header)
            for m in seg["metrics"]:
                sig = "*" if m["significant"] else " "
                print(
                    f"{sig} {m['metric_name']:<22} "
                    f"lift={m['relative_lift']:+.2%}  "
                    f"p={m['p_value']:.4f}  "
                    f"CI=[{m['ci_low']:+.4f}, {m['ci_high']:+.4f}]"
                )

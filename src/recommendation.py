"""
recommendation.py — Final SHIP / HOLD / REJECT / INVESTIGATE decision.

Integrates the outputs of every analysis module:
- frequentist.run_all_tests        
    — p-values + analytical CIs (user level)
- bayesian.run_all_bayesian        
    — P(T>C), credible intervals, expected loss
- segmentation.run_all_segmentation 
    — heterogeneity by device/country/source/medium
- guardrails.run_all_guardrails    
    — pass/warning/fail per policy threshold

Decision tree (first match wins):
  1. Guardrails FAIL                                 
    - HOLD
  2. Primary metric significantly NEGATIVE           
    - REJECT
  3. Segments show opposing significant effects      
    - INVESTIGATE
  4. Primary significant + P(T>C) > 0.95 + guards OK 
    - SHIP
  5. Otherwise                                       
    - HOLD

we do HOLD on guardrail FAIL instead of reject bc a guardrail breach 
means the feature isn't safe to ship AS-IS, not that the feature itself is bad.
"""

import pandas as pd

from frequentist import run_all_tests, PRIMARY_METRICS as FREQ_PRIMARY
from bayesian import run_all_bayesian, PROB_SHIP_THRESHOLD
from segmentation import run_all_segmentation
from guardrails import run_all_guardrails


VERDICT_SHIP = "SHIP"
VERDICT_HOLD = "HOLD"
VERDICT_REJECT = "REJECT"
VERDICT_INVESTIGATE = "INVESTIGATE"


def _primary_metric_names() -> list:
    return [name for name, _ in FREQ_PRIMARY]


def _significantly_negative(freq: dict, primary_names: list) -> list:
    """Primary metrics that are significant AND have negative lift."""
    return [
        f["metric_name"]
        for f in freq["primary"]
        if f["metric_name"] in primary_names
        and f["significant"]
        and f["absolute_lift"] < 0
    ]


def _significantly_positive_with_high_prob(
    freq: dict, bayes: dict, primary_names: list
) -> list:
    """Primary metrics with significant positive lift AND P(T>C) > threshold."""
    bayes_by_name = {r["metric_name"]: r for r in bayes["primary"]}
    winners = []
    for f in freq["primary"]:
        if f["metric_name"] not in primary_names:
            continue
        b = bayes_by_name.get(f["metric_name"])
        if (
            f["significant"]
            and f["absolute_lift"] > 0
            and b
            and b["prob_treatment_better"] > PROB_SHIP_THRESHOLD
        ):
            winners.append(f["metric_name"])
    return winners


def _heterogeneous_segments(seg_output: dict, primary_names: list) -> dict:
    """
    For each primary metric, return any segments where the effect was
    significant in OPPOSITE directions — i.e. treatment helps some users
    and hurts others. This is the INVESTIGATE signal -> i think this is
    known as simpsons paradox style heterogeneity. 
    """
    out = {}
    for metric in primary_names:
        pos, neg = [], []
        for seg_col, results in seg_output["segments"].items():
            for seg in results:
                for m in seg["metrics"]:
                    if m["metric_name"] != metric or not m["significant"]:
                        continue
                    label = f"{seg_col}={seg['segment_value']}"
                    if m["absolute_lift"] > 0:
                        pos.append(label)
                    elif m["absolute_lift"] < 0:
                        neg.append(label)
        if pos and neg:
            out[metric] = {"positive": pos, "negative": neg}
    return out


def make_recommendation(df: pd.DataFrame) -> dict:
    """Run every analysis, apply the decision tree, return verdict + summary."""
    freq = run_all_tests(df)
    bayes = run_all_bayesian(df)
    seg = run_all_segmentation(df)
    guards = run_all_guardrails(df)

    primary_names = _primary_metric_names()
    reasons = []

    if guards["overall_status"] == "fail":
        verdict = VERDICT_HOLD
        failed = [r["name"] for r in guards["results"] if r["status"] == "fail"]
        reasons.append(f"Guardrail FAIL on: {', '.join(failed)}")

    elif (neg := _significantly_negative(freq, primary_names)):
        verdict = VERDICT_REJECT
        reasons.append(
            f"Primary metric(s) significantly negative: {', '.join(neg)}"
        )

    elif (het := _heterogeneous_segments(seg, primary_names)):
        verdict = VERDICT_INVESTIGATE
        for metric, sides in het.items():
            reasons.append(
                f"{metric}: positive in [{', '.join(sides['positive'][:3])}], "
                f"negative in [{', '.join(sides['negative'][:3])}]"
            )

    elif (winners := _significantly_positive_with_high_prob(freq, bayes, primary_names)):
        verdict = VERDICT_SHIP
        reasons.append(
            f"Significant positive lift + P(T>C) > {PROB_SHIP_THRESHOLD:.2f} on: "
            f"{', '.join(winners)}"
        )
        if guards["overall_status"] == "warning":
            reasons.append("Guardrails show WARNING — review thresholds before rollout")

    else:
        verdict = VERDICT_HOLD
        reasons.append(
            "No primary metric reached both significance and posterior probability "
            f"> {PROB_SHIP_THRESHOLD:.2f}"
        )

    summary = _build_summary(verdict, reasons, freq, bayes, guards)

    return {
        "verdict": verdict,
        "reasons": reasons,
        "summary": summary,
        "frequentist": freq,
        "bayesian": bayes,
        "segmentation": seg,
        "guardrails": guards,
    }


def _build_summary(
    verdict: str, reasons: list, freq: dict, bayes: dict, guards: dict
) -> str:
    """Human-readable paragraph — PM/DS lead can paste into Slack or memo."""
    lines = [f"EXPERIMENT RECOMMENDATION: {verdict}", ""]

    lines.append("Primary metrics:")
    bayes_by_name = {r["metric_name"]: r for r in bayes["primary"]}
    for f in freq["primary"]:
        b = bayes_by_name.get(f["metric_name"], {})
        sig = "*" if f["significant"] else " "
        lines.append(
            f"{sig} {f['metric_name']:<22} "
            f"lift={f['relative_lift']:+.2%}  "
            f"p={f['p_value']:.4f}  "
            f"P(T>C)={b.get('prob_treatment_better', 0):.4f}"
        )

    lines.append("")
    lines.append(f"Guardrails: {guards['overall_status'].upper()}")
    for g in guards["results"]:
        lines.append(
            f"  - {g['name']:<22} {g['relative_diff']:+.2%}  ({g['status'].upper()})"
        )

    lines.append("")
    lines.append("Rationale:")
    for r in reasons:
        lines.append(f"  - {r}")

    return "\n".join(lines)


if __name__ == "__main__":
    from clean import build_clean_sessions
    from assign_experiment import assign_users
    from simulate_treatment import simulate_treatment_effects

    df = build_clean_sessions()
    df = assign_users(df)
    df = simulate_treatment_effects(df)

    rec = make_recommendation(df)
    print(rec["summary"])

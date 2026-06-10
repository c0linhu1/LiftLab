"""
5_Rollout_Memo.py — The paste-ready decision memo.

Combines the integrated verdict from /decision with a compact view of the
inputs that drove it (primary metrics, guardrails). Designed so a PM or
DS lead can copy the right-hand text block straight into Slack.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from api_client import decision, frequentist, bayesian, guardrails, banner_api_status


st.set_page_config(page_title="Rollout Memo", page_icon="📝", layout="wide")
st.title("Rollout Memo")
banner_api_status()
st.caption(
    "Final SHIP / HOLD / REJECT / INVESTIGATE call. Decision tree: guardrails first, "
    "then significantly-negative primary, then heterogeneous segments, then significant + "
    "high posterior, otherwise hold."
)

with st.spinner("Running full analysis (frequentist + Bayesian + segmentation + guardrails)..."):
    rec = decision()

verdict = rec["verdict"]
banner = {
    "SHIP": ("success", "🚢 Ship to all users."),
    "HOLD": ("warning", "⏸️ Hold. Iterate or extend before shipping."),
    "REJECT": ("error", "🛑 Do not ship. Primary metric regressed."),
    "INVESTIGATE": ("info", "🔬 Investigate. Effect is heterogeneous across segments."),
}.get(verdict, ("info", verdict))
getattr(st, banner[0])(f"**{verdict}** — {banner[1]}")

st.subheader("Rationale")
for r in rec["reasons"]:
    st.markdown(f"- {r}")

st.divider()

# Inputs that drove the decision
left, right = st.columns(2)

with left:
    st.markdown("### Primary metric snapshot")
    freq = frequentist()
    bayes = bayesian()
    bayes_by_name = {r["metric_name"]: r for r in bayes["primary"]}
    for f in freq["primary"]:
        b = bayes_by_name.get(f["metric_name"], {})
        sig_mark = "✅" if f["significant"] else "—"
        ship_mark = "🚢" if b.get("ship_recommended") else "⏸️"
        st.markdown(
            f"**{f['metric_name']}** {sig_mark} {ship_mark}  \n"
            f"Lift: `{f['relative_lift']:+.2%}`  ·  "
            f"p: `{f['p_value']:.4f}`  ·  "
            f"P(T>C): `{b.get('prob_treatment_better', 0):.4f}`  \n"
            f"95% CI: `[{f['ci_low']:+.4f}, {f['ci_high']:+.4f}]`"
        )

with right:
    st.markdown("### Guardrails")
    g = guardrails()
    badge = {"pass": "✅", "warning": "⚠️", "fail": "❌"}
    st.markdown(f"**Overall: {badge.get(g['overall_status'])} {g['overall_status'].upper()}**")
    for row in g["results"]:
        st.markdown(
            f"- **{row['name']}** {badge.get(row['status'])}  "
            f"`{row['relative_diff']:+.2%}` (limit `{row['fail_threshold']:.0%}`)"
        )

st.divider()

st.markdown("### Copy-paste summary")
st.code(rec["summary"], language=None)

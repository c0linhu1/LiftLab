"""
4_Guardrails.py — Policy threshold checks (independent of significance).

Direction-aware: a bounce-rate drop is good even though it's "below"
the control. Volatility going up is bad even though it's "higher" than
the control. Each guardrail spells out its direction in the table.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.express as px
import streamlit as st

from api_client import guardrails, banner_api_status


st.set_page_config(page_title="Guardrails", page_icon="🛡️", layout="wide")
st.title("Guardrails")
banner_api_status()
st.caption(
    "Threshold-based pass/warning/fail per metric. Independent of statistical "
    "significance — a metric can be statistically significant but still inside policy, "
    "and vice versa."
)

out = guardrails()
verdict = out["overall_status"]
verdict_colors = {"pass": "success", "warning": "warning", "fail": "error"}
getattr(st, verdict_colors.get(verdict, "info"))(
    f"**Overall: {verdict.upper()}**  ·  "
    f"{'rollout blocked' if verdict == 'fail' else 'review before rollout' if verdict == 'warning' else 'clear to ship on guardrails'}"
)

df = pd.DataFrame(out["results"])

# Status badges in the table
def _badge(s):
    return {"pass": "✅ PASS", "warning": "⚠️ WARNING", "fail": "❌ FAIL"}.get(s, s)

display = df.copy()
display["status"] = display["status"].map(_badge)
display = display[[
    "name", "direction", "control_value", "treatment_value",
    "relative_diff", "warning_threshold", "fail_threshold", "status",
]]
st.dataframe(display, use_container_width=True, hide_index=True)

# Visual: relative diff with thresholds
st.subheader("Relative change vs threshold")
plot_df = df.copy()
# Regression magnitude: positive = worse, regardless of direction
plot_df["regression"] = plot_df.apply(
    lambda r: r["relative_diff"] if r["direction"] == "lower_is_better" else -r["relative_diff"],
    axis=1,
)
fig = px.bar(
    plot_df.sort_values("regression"), x="regression", y="name", orientation="h",
    color="status",
    color_discrete_map={"pass": "#2ca02c", "warning": "#ffbf00", "fail": "#d62728"},
    labels={"regression": "Regression magnitude (positive = worse)", "name": ""},
)
# Add threshold lines per row using the max fail threshold shown
fig.add_vline(x=0, line_dash="dot", line_color="gray")
for thr, label in [
    (df["warning_threshold"].max(), "max warning"),
    (df["fail_threshold"].max(), "max fail"),
]:
    fig.add_vline(x=thr, line_dash="dash", line_color="orange",
                  annotation_text=label, annotation_position="top right")
fig.update_layout(xaxis_tickformat=".1%")
st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Regression magnitude is the relative change in the policy-bad direction. "
    "For bounce_rate and revenue_volatility, that's UP; for session_depth, that's DOWN. "
    "Bars to the right of the orange line have breached the fail threshold."
)

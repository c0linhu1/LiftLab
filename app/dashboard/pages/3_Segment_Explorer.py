"""
3_Segment_Explorer.py — Heterogeneous treatment effects.

Lets the user pick a segment dimension (device, country, source, medium)
and a primary metric, then renders the per-segment lift with CIs.
Underpowered segments are flagged but not hidden.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.express as px
import streamlit as st

from api_client import segments, banner_api_status


st.set_page_config(page_title="Segments", page_icon="🔬", layout="wide")
st.title("Segment Explorer")
banner_api_status()
st.caption(
    "Primary-metric tests run within each segment. Look for segments where the "
    "treatment effect concentrates — or, more interestingly, where it reverses."
)

with st.spinner("Running segment-level tests..."):
    out = segments()

st.markdown(
    f"**Min users per variant:** {out['min_users_per_variant']:,}  ·  "
    f"**Metrics tested:** {', '.join(f'`{m}`' for m in out['metrics_tested'])}"
)

segment_cols = list(out["segments"].keys())
selected_col = st.selectbox("Segment dimension", segment_cols)
selected_metric = st.selectbox("Primary metric", out["metrics_tested"])

rows = []
for seg in out["segments"][selected_col]:
    if seg["underpowered"]:
        rows.append({
            "segment": seg["segment_value"],
            "lift": None, "p_value": None,
            "ci_low": None, "ci_high": None,
            "n_ctrl": seg["n_users_control"], "n_treat": seg["n_users_treatment"],
            "significant": False, "underpowered": True,
        })
        continue
    for m in seg["metrics"]:
        if m["metric_name"] != selected_metric:
            continue
        rows.append({
            "segment": seg["segment_value"],
            "lift": m["relative_lift"],
            "p_value": m["p_value"],
            "ci_low": m["ci_low"], "ci_high": m["ci_high"],
            "n_ctrl": seg["n_users_control"], "n_treat": seg["n_users_treatment"],
            "significant": m["significant"], "underpowered": False,
        })

df = pd.DataFrame(rows).sort_values("lift", ascending=True, na_position="last")

st.subheader(f"{selected_metric} lift by {selected_col}")
plot_df = df.dropna(subset=["lift"])
fig = px.bar(
    plot_df, x="lift", y="segment", orientation="h",
    color="significant",
    color_discrete_map={True: "#2ca02c", False: "#cccccc"},
    labels={"lift": "Relative lift", "segment": ""},
)
fig.add_vline(x=0, line_dash="dash", line_color="gray")
fig.update_layout(xaxis_tickformat=".1%")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Full table")
st.dataframe(df, use_container_width=True, hide_index=True)

underpowered = df[df["underpowered"]]
if len(underpowered) > 0:
    st.warning(
        f"{len(underpowered)} segments below the {out['min_users_per_variant']}-user "
        "threshold — not tested. Their counts are shown but no lift / p-value."
    )

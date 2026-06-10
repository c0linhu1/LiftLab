"""
2_CUPED_Analysis.py — Raw vs CUPED-adjusted inference per metric.

Shows how much variance the pre-experiment covariates remove (or fail to
remove). Big variance reduction => narrower CIs / smaller p-values. On
this GA4 sample reductions are small because most users lack a real
pre-period; that finding is part of the story, not a bug.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from api_client import cuped, banner_api_status


st.set_page_config(page_title="CUPED Analysis", page_icon="📉", layout="wide")
st.title("CUPED Variance Reduction")
banner_api_status()
st.caption(
    "Y_adj = Y − θ(X − X̄), θ fit on combined data. Covariates are taken from "
    "each user's earliest experiment-window session."
)

with st.spinner("Computing CUPED..."):
    out = cuped()

st.markdown(
    f"**Users analyzed:** {out['n_users']:,}  ·  "
    f"**Covariates:** {', '.join(f'`{c}`' for c in out['covariates'])}"
)

df = pd.DataFrame(out["results"])

st.subheader("Raw vs adjusted, side-by-side")
display = df[[
    "metric_name",
    "raw_lift", "raw_p_value",
    "adj_lift", "adj_p_value",
    "variance_reduction", "ci_width_improvement",
]]
st.dataframe(display, use_container_width=True, hide_index=True)

# Variance reduction bar chart
st.subheader("Variance reduction per metric")
vr_df = df.sort_values("variance_reduction")
fig_vr = px.bar(
    vr_df, x="variance_reduction", y="metric_name", orientation="h",
    labels={"variance_reduction": "Variance reduction", "metric_name": ""},
    text=vr_df["variance_reduction"].map(lambda v: f"{v:.1%}"),
)
fig_vr.update_layout(xaxis_tickformat=".1%")
st.plotly_chart(fig_vr, use_container_width=True)

# CI width comparison
st.subheader("Confidence-interval width: raw vs CUPED-adjusted")
ci = pd.DataFrame({
    "metric": df["metric_name"],
    "Raw": df["raw_ci_high"] - df["raw_ci_low"],
    "CUPED-adjusted": df["adj_ci_high"] - df["adj_ci_low"],
}).melt("metric", var_name="series", value_name="width")
fig_ci = px.bar(ci, x="metric", y="width", color="series", barmode="group",
                labels={"width": "95% CI width on absolute lift"})
st.plotly_chart(fig_ci, use_container_width=True)

# Theta coefficients
st.subheader("θ coefficients (which covariate is doing the work)")
theta_df = pd.DataFrame(
    df["theta"].tolist(),
    columns=out["covariates"],
    index=df["metric_name"],
)
st.dataframe(theta_df, use_container_width=True)

st.info(
    "If variance reduction is small (<5%) across the board, the covariates "
    "aren't predicting the outcomes well — usually because most users in the "
    "experiment window lack pre-period history. CUPED is operational; the "
    "limit is the data."
)

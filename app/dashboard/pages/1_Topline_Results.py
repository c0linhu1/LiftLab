"""
1_Topline_Results.py — Frequentist + Bayesian side-by-side for primary,
secondary, and guardrail metrics. The reader can see when the two
frameworks agree (clean signal) and when they disagree (interesting).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.express as px
import streamlit as st

from api_client import frequentist, bayesian, banner_api_status


st.set_page_config(page_title="Topline Results", page_icon="📊", layout="wide")
st.title("Topline Results")
banner_api_status()
st.caption(
    "Frequentist (Welch's t / two-proportion z) and Bayesian (Beta-Binomial / "
    "Normal posterior) tests on user-level aggregates."
)

freq = frequentist()
bayes = bayesian()

def _merge(tier: str) -> pd.DataFrame:
    f = pd.DataFrame(freq[tier])
    b = pd.DataFrame(bayes[tier])[
        ["metric_name", "prob_treatment_better",
         "expected_loss_ship", "expected_loss_no_ship", "ship_recommended"]
    ]
    return f.merge(b, on="metric_name", how="left")


def _render_tier(tier: str, title: str) -> None:
    st.subheader(title)
    df = _merge(tier)

    cols = st.columns(min(len(df), 4))
    for i, row in df.iterrows():
        sig = "✅" if row["significant"] else "—"
        ship = "🚢" if row["ship_recommended"] else "⏸️"
        cols[i % len(cols)].metric(
            label=row["metric_name"],
            value=f"{row['relative_lift']:+.2%}",
            delta=f"p={row['p_value']:.4f}  P(T>C)={row['prob_treatment_better']:.3f}  {sig}{ship}",
            delta_color="off",
        )

    display = df[[
        "metric_name", "control_mean", "treatment_mean",
        "relative_lift", "p_value", "ci_low", "ci_high",
        "prob_treatment_better", "expected_loss_ship", "expected_loss_no_ship",
        "significant", "ship_recommended",
    ]]
    st.dataframe(display, use_container_width=True, hide_index=True)

    fig = px.bar(
        df, x="metric_name", y="relative_lift",
        error_y=df["ci_high"] - df["relative_lift"] * df["control_mean"],
        title=f"{title}: relative lift", labels={"relative_lift": "Relative lift"},
    )
    fig.update_layout(yaxis_tickformat=".1%")
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    st.plotly_chart(fig, use_container_width=True)


_render_tier("primary", "Primary metrics")
st.divider()
_render_tier("secondary", "Secondary metrics")
st.divider()
_render_tier("guardrail", "Guardrail metrics (raw — see Guardrails page for policy verdict)")

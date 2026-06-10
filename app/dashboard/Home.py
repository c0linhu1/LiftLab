import streamlit as st

from api_client import (
    list_experiments, get_experiment, srm, decision, banner_api_status,
)


st.set_page_config(page_title="LiftLab — Overview", page_icon="🧪", layout="wide")
st.title("LiftLab")
st.subheader("Cloud-native experimentation platform for product decisions")
banner_api_status()

st.divider()

# Active experiment 
experiments = list_experiments()
if not experiments:
    st.error("No experiments registered.")
    st.stop()

exp_summary = experiments[0]
exp = get_experiment(exp_summary["experiment_id"])

col1, col2 = st.columns([2, 1])
with col1:
    st.markdown(f"### Active experiment: **{exp['experiment_name']}**")
    st.markdown(f"_{exp['hypothesis']}_")
    st.markdown(
        f"**Primary metric:** `{exp['primary_metric']}`  ·  "
        f"**Guardrails:** {', '.join(f'`{g}`' for g in exp['guardrail_metrics'])}"
    )
    st.markdown(
        f"**Window:** {exp['start_date']} → {exp['end_date']}  ·  "
        f"**Split:** {exp['treatment_split']:.0%} treatment  ·  "
        f"**Stratified by:** {', '.join(exp['stratify_by']) if exp['stratify_by'] else 'none'}"
    )

with col2:
    st.metric("Users in experiment", f"{exp['n_users']:,}")
    st.metric("Sessions in experiment", f"{exp['n_sessions']:,}")

st.divider()

# SRM + Decision side-by-side
left, right = st.columns(2)

with left:
    st.markdown("### Sample-ratio mismatch (SRM)")
    s = srm()
    if s["passed"]:
        st.success(
            f"PASS — chi² = {s['chi2']:.4f}, p = {s['p_value']:.4f}\n\n"
            f"Treatment share: {s['actual_treatment_share']:.4f}  "
            f"(expected {s['expected_treatment_share']:.2f})"
        )
    else:
        st.error(
            f"FAIL — chi² = {s['chi2']:.4f}, p = {s['p_value']:.4f}\n\n"
            "Variant split deviates from the configured ratio. "
            "Downstream metric results are NOT trustworthy."
        )

with right:
    st.markdown("### Final verdict")
    with st.spinner("Running full analysis..."):
        rec = decision()
    verdict = rec["verdict"]
    colormap = {
        "SHIP": "success", "HOLD": "warning",
        "REJECT": "error", "INVESTIGATE": "info",
    }
    getattr(st, colormap.get(verdict, "info"))(f"**{verdict}**")
    for reason in rec["reasons"]:
        st.markdown(f"- {reason}")

st.divider()
st.caption("Use the sidebar to drill into topline results, CUPED, segments, guardrails, and the full memo.")

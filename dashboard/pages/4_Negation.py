from __future__ import annotations

import streamlit as st

from utils import data_loader
from utils.charts import Chart, negation_rules_bar


def _caption(chart: Chart) -> None:
    st.plotly_chart(chart.fig, use_container_width=True)
    st.caption(chart.caption)

st.title("Negation Analysis")
st.markdown(
    "Negation counterfactuals reveal a clinically important failure mode. We insert explicit negation into common ADE causal "
    "constructions (e.g., “reported” → “did not report”) and measure whether the prediction flips. Low flip rates indicate the "
    "model often ignores negation in that pattern — a direct patient-safety concern."
)

neg = data_loader.load_negation_analysis()
rules_df = data_loader.negation_rule_frame(neg)
sent_df = data_loader.negation_per_sentence_frame(neg)

if rules_df is None or len(rules_df) == 0:
    st.warning("Missing `negation_analysis.json` (or empty content), so this page cannot be shown.")
    st.stop()

st.subheader("Flip rate by negation pattern")
chart = negation_rules_bar(
    rules_df,
    title="Flip rates across 10 negation patterns",
    caption="Bars are teal if flip rate is ≥ 50% (model responds to negation) and coral if < 50% (model often ignores negation). "
    "The worst pattern is typically “reported”, where only ~7% of cases flip.",
    threshold=0.5,
)
_caption(chart)

flip_rate = float(neg.get("flip_rate", 0.0)) if isinstance(neg, dict) else 0.0
unaffected_pct = (1.0 - flip_rate) * 100.0
st.metric("Negation ignored (unaffected by negation)", f"{unaffected_pct:.1f}%")
st.caption(
    "Clinical safety concern: 42.8% of negation-edited sentences are unaffected. This is computed as 1 − overall negation flip rate."
)

st.subheader("Concrete example (worst-performing pattern)")
st.markdown(
    "We highlight a representative sentence from the lowest-flip pattern (often `reported`). "
    "This is a concrete example of the model appearing to ignore negation."
)

if sent_df is None or len(sent_df) == 0:
    st.info("No per-sentence negation examples found.")
else:
    # Find the worst rule by flip rate.
    worst = rules_df.iloc[0]
    worst_pattern = worst["pattern"]

    # Rows store the original regex; our frame strips \\b and parentheses, so match loosely.
    cand = sent_df.copy()
    if "rule_used" in cand.columns:
        cand = cand[cand["rule_used"].astype(str).str.contains(worst_pattern, case=False, na=False)]
    row = cand.head(1)
    if len(row) == 0:
        row = sent_df.head(1)

    r = row.iloc[0]
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Original sentence")
        st.write(r.get("original_text", "—"))
        st.metric("Prediction (orig)", "ADE" if int(r.get("original_pred", 0)) == 1 else "non-ADE")
        st.metric("P(ADE) (orig)", f"{float(r.get('original_prob', 0.0)):.4f}")
    with c2:
        st.markdown("#### Negated sentence")
        st.write(r.get("negated_text", "—"))
        st.metric("Prediction (negated)", "ADE" if int(r.get("negated_pred", 0)) == 1 else "non-ADE")
        st.metric("P(ADE) (negated)", f"{float(r.get('negated_prob', 0.0)):.4f}")

    st.caption(
        f"Example selected from the lowest-flip negation pattern bucket. Bucket label: `{worst_pattern}`. "
        "This illustrates how adding negation can fail to change the prediction."
    )


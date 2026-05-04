from __future__ import annotations

import pandas as pd
import streamlit as st

from utils import data_loader
from utils.charts import (
    Chart,
    bucket_flip_rate_bar,
    interaction_direction_pie,
    minimality_distribution_bar,
)


def _caption(chart: Chart) -> None:
    st.plotly_chart(chart.fig, use_container_width=True)
    st.caption(chart.caption)

st.title("Error Patterns")
st.markdown(
    "To rule out trivial confounders, we segment flip rates by sentence properties. The key finding is that drug masking flip rates "
    "remain high and largely flat across sentence length, number of drug mentions, and causal marker presence — more context does not "
    "reduce drug-token reliance. We also summarize minimality and SHAP interaction direction."
)

err = data_loader.load_error_patterns()
frames = data_loader.error_patterns_frames(err)
if not frames:
    st.warning("Missing `error_patterns.json`, so the stratified flip-rate charts cannot be shown.")
else:
    c1, c2, c3 = st.columns(3)
    with c1:
        df = frames.get("by_sentence_length", pd.DataFrame())
        if len(df) > 0:
            chart = bucket_flip_rate_bar(
                df,
                title="Flip rate by sentence length bucket",
                caption="Family A drug-masking flip rate is flat across sentence lengths. Longer sentences with more context do not reduce drug reliance.",
            )
            _caption(chart)
        else:
            st.info("No sentence-length stratification available.")

    with c2:
        df = frames.get("by_drug_mention_count", pd.DataFrame())
        if len(df) > 0:
            chart = bucket_flip_rate_bar(
                df,
                title="Flip rate by drug-mention count",
                caption="Flip rate stays >71% even when multiple drug mentions are present, indicating redundancy but persistent drug dependence.",
            )
            _caption(chart)
        else:
            st.info("No drug-count stratification available.")

    with c3:
        df = frames.get("by_causal_marker", pd.DataFrame())
        if len(df) > 0:
            chart = bucket_flip_rate_bar(
                df,
                title="Flip rate by causal-marker presence",
                caption="Flip rates are similar with and without causal markers, reinforcing that drug surface tokens drive predictions beyond explicit causal phrasing.",
            )
            _caption(chart)
        else:
            st.info("No causal-marker stratification available.")

st.divider()

st.subheader("Minimality: how many drugs are needed to flip?")
min_payload = data_loader.load_minimality()
min_dist = data_loader.minimality_distribution(min_payload)
if min_dist is None or len(min_dist) == 0:
    st.warning("Missing `minimality_analysis.json`, so the minimality distribution cannot be shown.")
else:
    chart = minimality_distribution_bar(
        min_dist,
        title="Minimum number of drugs to mask to flip the prediction",
        caption="Minimality analysis shows 82% of sentences flip with just one drug token masked — the model’s drug reliance is highly concentrated, not distributed.",
    )
    _caption(chart)

st.subheader("SHAP interaction direction (cooperative vs competitive)")
shap_int = data_loader.load_shap_interactions()
rates = data_loader.shap_interaction_direction_rates(shap_int)
if rates is None:
    st.warning("Missing `shap_interactions.json`, so the interaction direction summary cannot be shown.")
else:
    chart = interaction_direction_pie(
        rates,
        title="Interaction directions across SHAP interaction samples",
        caption="SHAP interaction analysis finds drug and context tokens act cooperatively in 88.6% of cases (competitive in 11.4%).",
    )
    _caption(chart)


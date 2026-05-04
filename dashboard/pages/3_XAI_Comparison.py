from __future__ import annotations

import math

import pandas as pd
import streamlit as st

from utils import data_loader
from utils.charts import Chart, xai_ratio_bar, xai_top3_bar


def _caption(chart: Chart) -> None:
    st.plotly_chart(chart.fig, use_container_width=True)
    st.caption(chart.caption)

st.title("XAI Method Comparison")
st.markdown(
    "We compare five explanation approaches on the same ADE-positive sentences: counterfactual perturbations (causal ground truth), "
    "SHAP, LIME, Integrated Gradients (IG), and attention visualization. Because these methods answer different questions, they can "
    "disagree — that disagreement is itself diagnostic of faithfulness."
)

stats = data_loader.load_statistical_tests()
sr = data_loader.get_sensitivity_ratio(stats)

att = data_loader.load_attention_summary()
att_ratio = None
if isinstance(att, dict):
    agg = att.get("aggregate", {})
    if isinstance(agg, dict) and "mean_drug_nondrug_ratio" in agg:
        att_ratio = float(agg["mean_drug_nondrug_ratio"])

shap_full = data_loader.load_shap_full()
lime_full = data_loader.load_lime_full()
ig_full = data_loader.load_ig_full()

def _ratio(payload):
    row = data_loader.xai_aggregate_row(payload, "x")
    return None if row is None else row.get("ratio")

def _pct(payload):
    row = data_loader.xai_aggregate_row(payload, "x")
    return None if row is None else row.get("pct_top3")


ratio_df = pd.DataFrame(
    [
        {"method": "Attention", "ratio": att_ratio if att_ratio is not None else math.nan},
        {"method": "SHAP", "ratio": float(_ratio(shap_full)) if _ratio(shap_full) is not None else math.nan},
        {"method": "Integrated Gradients", "ratio": float(_ratio(ig_full)) if _ratio(ig_full) is not None else math.nan},
        {"method": "Counterfactual (Flip Rate)", "ratio": float(sr.point_estimate) if sr else math.nan},
        {"method": "LIME", "ratio": float(_ratio(lime_full)) if _ratio(lime_full) is not None else math.nan},
    ]
)

st.subheader("Drug token importance across methods")
chart = xai_ratio_bar(
    ratio_df,
    title="Drug / non-drug token importance ratio by method",
    caption="LIME confirms the strongest drug signal (2.87×), while SHAP (1.23×) and IG (1.54×) also emphasize drug tokens. "
    "Counterfactuals provide the causal baseline via the Sensitivity Ratio (2.46×). Attention contradicts all other methods (0.59×).",
)
_caption(chart)

st.warning(
    "Attention is the outlier: it contradicts all other methods. This replicates a well-known finding in NLP: "
    "attention weights are not a faithful proxy for causal importance."
)

st.subheader("Drug appears in Top-3 attributions")
top3_df = pd.DataFrame(
    [
        {"method": "SHAP", "pct_top3": float(_pct(shap_full)) * 100.0 if _pct(shap_full) is not None else math.nan},
        {"method": "LIME", "pct_top3": float(_pct(lime_full)) * 100.0 if _pct(lime_full) is not None else math.nan},
        {"method": "IG", "pct_top3": float(_pct(ig_full)) * 100.0 if _pct(ig_full) is not None else math.nan},
    ]
)
chart = xai_top3_bar(
    top3_df,
    title="How often the drug token is in the top-3 explanations",
    caption="Across attribution methods, drug tokens frequently appear among the most important features: SHAP 67.9%, "
    "IG 80.3%, and LIME 95.3% (Top-3 coverage).",
)
_caption(chart)

st.subheader("Why methods differ (plain English)")
st.markdown(
    "Counterfactuals measure *causation*: if we mask a drug mention, does the prediction change? "
    "SHAP and LIME estimate *token importance* by probing the model under perturbations; IG follows gradients through the network. "
    "Attention is best viewed as a routing / alignment signal, not a faithful explanation. "
    "Our multi-method comparison uses counterfactuals as the causal ground truth against which correlational methods are evaluated."
)

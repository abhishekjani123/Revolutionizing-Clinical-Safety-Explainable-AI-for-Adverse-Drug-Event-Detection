from __future__ import annotations
import streamlit as st

from utils import data_loader
from utils.charts import (
    COLORS,
    Chart,
    confidence_delta_histograms,
    failure_modes_bar,
    sensitivity_ratio_ci,
)


def _section_intro(text: str) -> None:
    st.markdown(text)


def _caption(chart: Chart) -> None:
    st.plotly_chart(chart.fig, use_container_width=True)
    st.caption(chart.caption)

st.title("Counterfactual Analysis")
_section_intro(
    "Counterfactuals directly test *causal influence* by applying minimal, targeted edits and checking whether the model’s "
    "prediction changes. We operationalize this at population scale using the flip-rate metric across three families: "
    "Family A (drug-name masking), Family B (brand/generic substitution via RxNorm), and Family C (symptom-term masking)."
)

stats = data_loader.load_statistical_tests()
sr = data_loader.get_sensitivity_ratio(stats)

st.subheader("Sensitivity Ratio (Family A / Family C)")
if sr:
    chart = sensitivity_ratio_ci(
        point=sr.point_estimate,
        low=sr.ci_95_low,
        high=sr.ci_95_high,
        title="Drug sensitivity is ~2.46× symptom sensitivity",
        caption="The sensitivity ratio compares flip rates when masking drugs (Family A) versus masking symptoms (Family C). "
        "Error bars show the 95% bootstrap confidence interval reported in `statistical_tests.json` (paired McNemar test is highly significant).",
    )
    _caption(chart)
else:
    st.warning("Missing `statistical_tests.json`, so the sensitivity ratio chart cannot be shown.")

st.subheader("Confidence delta distributions")
fa = data_loader.load_family_a_sentences()
fb = data_loader.load_family_b_sentences()
fc = data_loader.load_family_c_sentences()

series_by_name = {}
if fa is not None and "confidence_delta" in fa.columns:
    series_by_name["Family A (Drug Masking)"] = fa["confidence_delta"]
if fb is not None and "confidence_delta" in fb.columns:
    series_by_name["Family B (Brand Substitution)"] = fb["confidence_delta"]
if fc is not None and "confidence_delta" in fc.columns:
    series_by_name["Family C (Symptom Masking)"] = fc["confidence_delta"]

if series_by_name:
    chart = confidence_delta_histograms(
        series_by_name,
        title="How much confidence drops after each perturbation",
        caption="Confidence deltas complement binary flip rate: large deltas indicate strong sensitivity even when the label does not change. "
        "Drug masking produces the largest mean confidence drop, consistent with drug-token reliance.",
    )
    _caption(chart)
else:
    st.warning("Missing family JSONs with `confidence_delta`, so the histogram cannot be shown.")

st.subheader("Paired failure modes (Family A vs Family C)")
err = data_loader.load_error_patterns()
pfm = data_loader.paired_failure_modes(err)
if pfm:
    chart = failure_modes_bar(
        pfm,
        title="Which sentences flip under drug masking vs symptom masking",
        caption="This breakdown is computed on the paired subset of sentences present in both Family A and Family C (n paired). "
        "It quantifies asymmetric sensitivity: many sentences flip under drug masking but not under symptom masking.",
    )
    _caption(chart)
else:
    st.warning("Missing `error_patterns.json` (or missing `paired_failure_modes`), so this chart cannot be shown.")

st.divider()
st.subheader("Sentence explorer")
_section_intro(
    "Pick a sentence ID (stable across experiments) to see the original sentence, its perturbed version, "
    "and the model's prediction probabilities side-by-side."
)

if fa is None and fb is None and fc is None:
    st.info("Family result files are missing, so the sentence explorer is unavailable.")
else:
    # Build a unified list of IDs from whichever families are available.
    ids = []
    for df in [fa, fb, fc]:
        if df is not None and "sentence_id" in df.columns:
            ids.extend(df["sentence_id"].astype(str).tolist())
    ids = sorted(list(dict.fromkeys(ids)))

    if not ids:
        st.info("No `sentence_id` values found in the family JSONs.")
    else:
        query = st.text_input("Filter by text (optional)", value="")
        candidate_ids = ids
        if query and fa is not None and "original_text" in fa.columns:
            mask = fa["original_text"].str.contains(query, case=False, na=False)
            candidate_ids = fa.loc[mask, "sentence_id"].astype(str).head(200).tolist() or ids

        sid = st.selectbox("Choose a sentence_id", options=candidate_ids, index=0)

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown("### Family A — Drug masking")
            if fa is None:
                st.caption("Missing `family_a_with_confidence.json`.")
            else:
                row = fa.loc[fa["sentence_id"].astype(str) == str(sid)].head(1)
                if len(row) == 0:
                    st.caption("No record for this ID in Family A.")
                else:
                    r = row.iloc[0]
                    st.write("**Original**:", r.get("original_text", "—"))
                    st.write("**Masked**:", r.get("masked_text", "—"))
                    st.metric("P(ADE) original", f"{float(r.get('original_prob', 0.0)):.4f}")
                    st.metric("P(ADE) masked", f"{float(r.get('masked_prob', 0.0)):.4f}")
                    st.write("**Flipped**:", bool(r.get("flipped", False)))

        with col_b:
            st.markdown("### Family B — Brand substitution")
            if fb is None:
                st.caption("Missing `family_b_with_confidence.json`.")
            else:
                row = fb.loc[fb["sentence_id"].astype(str) == str(sid)].head(1)
                if len(row) == 0:
                    st.caption("No record for this ID in Family B.")
                else:
                    r = row.iloc[0]
                    st.write("**Original**:", r.get("original_text", "—"))
                    st.write("**Brand swapped**:", r.get("replaced_text", "—"))
                    st.metric("P(ADE) original", f"{float(r.get('original_prob', 0.0)):.4f}")
                    st.metric("P(ADE) swapped", f"{float(r.get('replaced_prob', 0.0)):.4f}")
                    st.write("**Flipped**:", bool(r.get("flipped", False)))

        with col_c:
            st.markdown("### Family C — Symptom masking")
            if fc is None:
                st.caption("Missing `family_c_with_confidence.json`.")
            else:
                row = fc.loc[fc["sentence_id"].astype(str) == str(sid)].head(1)
                if len(row) == 0:
                    st.caption("No record for this ID in Family C.")
                else:
                    r = row.iloc[0]
                    st.write("**Original**:", r.get("original_text", "—"))
                    st.write("**Masked**:", r.get("masked_text", "—"))
                    st.metric("P(ADE) original", f"{float(r.get('original_prob', 0.0)):.4f}")
                    st.metric("P(ADE) masked", f"{float(r.get('masked_prob', 0.0)):.4f}")
                    st.write("**Flipped**:", bool(r.get("flipped", False)))


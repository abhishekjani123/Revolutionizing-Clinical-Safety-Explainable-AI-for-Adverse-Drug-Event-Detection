from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import streamlit as st

from utils import data_loader
from utils.charts import COLORS, Chart, flip_rate_comparison_bar


REPO_URL = "https://github.com/abhishekjani123/Revolutionizing-Clinical-Safety-Explainable-AI-for-Adverse-Drug-Event-Detection"
NOTEBOOK_URL = (
    "https://github.com/abhishekjani123/Revolutionizing-Clinical-Safety-Explainable-AI-for-Adverse-Drug-Event-Detection"
    "/blob/main/Extending_ADE_detection_with_Explainable_Counterfactuals_and_BioBERT_Final.ipynb"
)
REPORT_URL = (
    "https://github.com/abhishekjani123/Revolutionizing-Clinical-Safety-Explainable-AI-for-Adverse-Drug-Event-Detection"
    "/blob/main/ADE_Report_Extended.pdf"
)
SLIDES_RAW_URL = (
    "https://github.com/abhishekjani123/Revolutionizing-Clinical-Safety-Explainable-AI-for-Adverse-Drug-Event-Detection"
    "/raw/main/Final.pptx"
)


def _kpi_card(*, label: str, value: str, help_text: str) -> None:
    st.markdown(
        f"""
<div style="background: white; border: 1px solid {COLORS['mid_gray']}; border-radius: 12px; padding: 14px;">
  <div style="font-size: 13px; color: {COLORS['dark_gray']}; margin-bottom: 6px;">{label}</div>
  <div style="font-size: 28px; font-weight: 700; color: {COLORS['navy']}; line-height: 1.1;">{value}</div>
  <div style="font-size: 12px; color: {COLORS['dark_gray']}; margin-top: 8px;">{help_text}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def _pct(x: float) -> float:
    return float(x) * 100.0


def render_overview() -> None:
    st.title("Are Drug Mentions What Really Drive ADE Predictions?")

    st.markdown(
        "Adverse Drug Events (ADEs) are a major clinical safety concern, and detecting them from clinical text is both "
        "important and challenging. Although fine-tuned biomedical language models like BioBERT achieve strong benchmark "
        "performance, their lack of interpretability hinders real-world adoption. Here we present a *multi-method* XAI "
        "analysis using counterfactual perturbation (three families), SHAP, LIME, Integrated Gradients (IG), attention "
        "visualization, negation counterfactuals, and minimality analysis."
    )

    stats = data_loader.load_statistical_tests()
    sr = data_loader.get_sensitivity_ratio(stats)

    fam_a = data_loader.load_family_a()
    fam_a_flip = fam_a.get("flip_rate") if isinstance(fam_a, dict) else None

    neg = data_loader.load_negation_analysis()
    neg_flip = neg.get("flip_rate") if isinstance(neg, dict) else None
    neg_unaffected = (1.0 - float(neg_flip)) if neg_flip is not None else None

    c1, c2, c3 = st.columns(3)
    with c1:
        if sr:
            _kpi_card(
                label="Sensitivity Ratio (Drug vs Symptom)",
                value=f"{sr.point_estimate:.2f}×",
                help_text=f"95% CI: [{sr.ci_95_low:.2f}, {sr.ci_95_high:.2f}] (McNemar p ≪ 0.001).",
            )
        else:
            _kpi_card(
                label="Sensitivity Ratio (Drug vs Symptom)",
                value="—",
                help_text="Missing `statistical_tests.json`.",
            )

    with c2:
        if fam_a_flip is not None:
            _kpi_card(
                label="Drug mask flip rate (Family A, ADE)",
                value=f"{_pct(float(fam_a_flip)):.1f}%",
                help_text="Percent of ADE-positive sentences whose prediction flips when drug mentions are masked.",
            )
        else:
            _kpi_card(
                label="Drug mask flip rate (Family A, ADE)",
                value="—",
                help_text="Missing `family_a_with_confidence.json`.",
            )

    with c3:
        if neg_unaffected is not None:
            _kpi_card(
                label="Negation ignored",
                value=f"{_pct(float(neg_unaffected)):.1f}%",
                help_text="Percent of negation counterfactuals that do not change the model prediction.",
            )
        else:
            _kpi_card(
                label="Negation ignored",
                value="—",
                help_text="Missing `negation_analysis.json`.",
            )

    st.divider()

    st.subheader("Flip rates across counterfactual families")
    st.markdown(
        "Flip rate is the percentage of sentences whose predicted label changes after the perturbation. "
        "Family A masks drug mentions; Family B swaps generic drugs for brand names; Family C masks symptoms."
    )

    missing = data_loader.missing_files(
        [
            "family_a_with_confidence.json",
            "family_b_with_confidence.json",
            "family_c_with_confidence.json",
            "masking_non_ade_flips.json",
            "masking_non_ade_nonflips.json",
        ]
    )
    if missing:
        st.warning("Some files are missing in `dashboard/results/`: " + ", ".join(missing))

    fam_b = data_loader.load_family_b()
    fam_c = data_loader.load_family_c()
    nonade_counts = data_loader.load_family_a_non_ade_counts()

    rows: list[dict[str, float | str]] = []
    if isinstance(fam_a, dict) and fam_a.get("flip_rate") is not None:
        rows.append({"label": "Family A (Drug Mask) — ADE", "flip_rate_pct": _pct(float(fam_a["flip_rate"]))})

    if nonade_counts and nonade_counts.get("total", 0) > 0:
        nonade_rate = nonade_counts["flipped"] / nonade_counts["total"]
        rows.append({"label": "Family A (Drug Mask) — non-ADE", "flip_rate_pct": _pct(nonade_rate)})

    if isinstance(fam_b, dict) and fam_b.get("flip_rate") is not None:
        rows.append({"label": "Family B (Brand Sub) — ADE", "flip_rate_pct": _pct(float(fam_b["flip_rate"]))})

    if isinstance(fam_c, dict) and fam_c.get("flip_rate") is not None:
        rows.append({"label": "Family C (Symptom Mask) — ADE", "flip_rate_pct": _pct(float(fam_c["flip_rate"]))})

    if not rows:
        st.info("No flip-rate data found yet. Ensure JSON files exist in `dashboard/results/`.")
        return

    df = pd.DataFrame(rows)
    chart = flip_rate_comparison_bar(
        df,
        title="Prediction flip rates across perturbation families",
        caption=(
            "Drug-name masking flips 74.5% of ADE-positive predictions, while symptom-term masking on the same sentences "
            "flips only 30.2%, yielding a sensitivity ratio of 2.46. This is the causal ground-truth signal that we compare "
            "against correlational attribution methods."
        ),
    )
    st.plotly_chart(chart.fig, use_container_width=True)
    st.caption(chart.caption)

    st.divider()
    st.subheader("Project links and team")
    st.markdown(
        "Below are the canonical project artifacts (repo, notebook, report, slides) and the contributor profiles."
    )

    c_links, c_team = st.columns([1.2, 1.8])
    with c_links:
        st.markdown("#### Links")
        st.link_button("GitHub repo", REPO_URL, use_container_width=True)
        st.link_button("Final notebook", NOTEBOOK_URL, use_container_width=True)
        st.link_button("Final report (PDF)", REPORT_URL, use_container_width=True)
        st.link_button("Slides (PPTX)", SLIDES_RAW_URL, use_container_width=True)

    with c_team:
        st.markdown("#### Team")
        members = [
            {
                "name": "Abhishek Jani",
                "handle": "abhishekjani123",
                "role": "",
            },
            {
                "name": "Shrey Patel",
                "handle": "patelshrey40",
                "role": "",
            },
            {
                "name": "Mustafa Adil",
                "handle": "madil786110",
                "role": "",
            },
        ]

        cols = st.columns(3, gap="large")
        for col, m in zip(cols, members):
            profile = f"https://github.com/{m['handle']}"
            avatar = f"https://github.com/{m['handle']}.png?size=160"
            with col:
                st.markdown(
                    f"""
<div class="team-card">
  <div class="team-avatar">
    <img src="{avatar}" alt="{m['name']}" />
  </div>
  <div class="team-name"><a href="{profile}" target="_blank" rel="noopener noreferrer">{m['name']}</a></div>
  <div class="team-handle">@{m['handle']}</div>
</div>
""",
                    unsafe_allow_html=True,
                )


def main() -> None:
    st.set_page_config(page_title="ADE XAI Dashboard", layout="wide", initial_sidebar_state="expanded")

    st.markdown(
        f"""
<style>
  .stApp {{ background-color: {COLORS['light_gray']}; }}
  section[data-testid="stSidebar"] > div {{ background-color: white; }}
  /* Improve vertical rhythm so headers/charts don’t visually merge */
  .block-container {{ padding-top: 2rem; padding-bottom: 2.5rem; }}
  h1 {{ margin: 0.25rem 0 1rem 0; }}
  h2 {{ margin: 1.4rem 0 0.6rem 0; }}
  h3 {{ margin: 1.1rem 0 0.5rem 0; }}
  [data-testid="stMarkdownContainer"] p {{ margin-bottom: 0.9rem; }}
  /* Team cards */
  .team-card {{
    background: white;
    border: 1px solid {COLORS['mid_gray']};
    border-radius: 14px;
    padding: 16px 14px;
    text-align: center;
    min-height: 210px;
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
    gap: 10px;
  }}
  .team-avatar {{
    width: 92px;
    height: 92px;
    margin: 0 auto;
    border-radius: 16px;
    overflow: hidden;
    border: 1px solid {COLORS['mid_gray']};
    background: {COLORS['light_gray']};
  }}
  .team-avatar img {{
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
  }}
  .team-name a {{
    color: {COLORS['navy']};
    font-weight: 750;
    font-size: 18px;
    text-decoration: none;
  }}
  .team-name a:hover {{ text-decoration: underline; }}
  .team-handle {{
    color: {COLORS['dark_gray']};
    font-size: 13px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }}
</style>
""",
        unsafe_allow_html=True,
    )

    # Avoid duplicate \"Home\" vs \"Overview\" pages.
    # Home becomes a launcher that routes to the real Overview page.
    try:
        st.switch_page("pages/1_Overview.py")
    except Exception:
        st.title("ADE XAI Dashboard")
        st.markdown("Use the sidebar to open **Overview** and other pages.")


if __name__ == "__main__":
    main()


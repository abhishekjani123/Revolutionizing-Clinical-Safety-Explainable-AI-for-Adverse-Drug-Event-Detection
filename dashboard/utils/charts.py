from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import plotly.graph_objects as go


COLORS = {
    "teal": "#2A9D8F",
    "coral": "#E76F51",
    "navy": "#264653",
    "light_gray": "#F3F4F6",
    "mid_gray": "#E5E7EB",
    "dark_gray": "#374151",
}


@dataclass(frozen=True)
class Chart:
    fig: go.Figure
    caption: str


def _pct(x: float) -> float:
    return float(x) * 100.0


def flip_rate_comparison_bar(df: pd.DataFrame, *, title: str, caption: str) -> Chart:
    """
    Expects columns: label, flip_rate_pct
    """
    fig = go.Figure(
        data=[
            go.Bar(
                x=df["label"],
                y=df["flip_rate_pct"],
                marker_color=COLORS["teal"],
                text=[f"{v:.1f}%" for v in df["flip_rate_pct"]],
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        title=title,
        xaxis_title="Experiment",
        yaxis_title="Flip rate (%)",
        yaxis=dict(range=[0, max(5, float(df["flip_rate_pct"].max()) * 1.25)]),
        margin=dict(l=20, r=20, t=60, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return Chart(fig=fig, caption=caption)


def sensitivity_ratio_ci(*, point: float, low: float, high: float, title: str, caption: str) -> Chart:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[point],
            y=["Sensitivity Ratio (A/C)"],
            mode="markers",
            marker=dict(size=14, color=COLORS["navy"]),
            error_x=dict(type="data", symmetric=False, array=[high - point], arrayminus=[point - low]),
            hovertemplate="Sensitivity Ratio: %{x:.3f}×<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Ratio (×)",
        yaxis_title="",
        margin=dict(l=20, r=20, t=60, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return Chart(fig=fig, caption=caption)


def confidence_delta_histograms(series_by_name: dict[str, pd.Series], *, title: str, caption: str, nbins: int = 60) -> Chart:
    palette = {
        "Family A (Drug Masking)": COLORS["teal"],
        "Family B (Brand Substitution)": COLORS["coral"],
        "Family C (Symptom Masking)": COLORS["navy"],
    }

    fig = go.Figure()
    for name, s in series_by_name.items():
        if s is None or len(s) == 0:
            continue
        fig.add_trace(
            go.Histogram(
                x=s,
                name=name,
                opacity=0.55,
                nbinsx=nbins,
                marker_color=palette.get(name, COLORS["dark_gray"]),
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="Confidence delta (original_prob − perturbed_prob)",
        yaxis_title="Count",
        barmode="overlay",
        margin=dict(l=20, r=20, t=60, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return Chart(fig=fig, caption=caption)


def failure_modes_bar(counts: dict[str, int], *, title: str, caption: str) -> Chart:
    order = [
        ("both_flipped", "Both flipped (A & C)"),
        ("only_a", "Only A flipped"),
        ("only_c", "Only C flipped"),
        ("neither", "Neither flipped"),
    ]
    x = [label for _, label in order]
    y = [int(counts.get(k, 0)) for k, _ in order]
    fig = go.Figure(
        data=[
            go.Bar(
                x=x,
                y=y,
                marker_color=[COLORS["navy"], COLORS["teal"], COLORS["coral"], COLORS["mid_gray"]],
                text=y,
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        title=title,
        xaxis_title="Paired failure mode (Family A vs Family C)",
        yaxis_title="Number of sentences",
        margin=dict(l=20, r=20, t=60, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return Chart(fig=fig, caption=caption)


def xai_ratio_bar(df: pd.DataFrame, *, title: str, caption: str) -> Chart:
    """
    Expects columns: method, ratio
    """
    fig = go.Figure(
        data=[
            go.Bar(
                x=df["method"],
                y=df["ratio"],
                marker_color=[COLORS["mid_gray"], COLORS["teal"], COLORS["teal"], COLORS["navy"], COLORS["coral"]],
                text=[f"{v:.2f}×" for v in df["ratio"]],
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        title=title,
        xaxis_title="Method",
        yaxis_title="Drug / Non-drug token importance ratio (×)",
        margin=dict(l=20, r=20, t=60, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(range=[0, max(1.0, float(df["ratio"].max()) * 1.25)]),
    )
    fig.add_hline(y=1.0, line_dash="dash", line_color=COLORS["coral"], annotation_text="No preference (1.0)")
    fig.add_hline(y=2.0, line_dash="dot", line_color=COLORS["dark_gray"], annotation_text="Success threshold (2.0)")
    return Chart(fig=fig, caption=caption)


def xai_top3_bar(df: pd.DataFrame, *, title: str, caption: str) -> Chart:
    """
    Expects columns: method, pct_top3
    """
    fig = go.Figure(
        data=[
            go.Bar(
                x=df["method"],
                y=df["pct_top3"],
                marker_color=[COLORS["teal"], COLORS["coral"], COLORS["teal"]],
                text=[f"{v:.1f}%" for v in df["pct_top3"]],
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        title=title,
        xaxis_title="Method",
        yaxis_title="Drug token appears in Top-3 attributions (%)",
        margin=dict(l=20, r=20, t=60, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(range=[0, 105]),
    )
    return Chart(fig=fig, caption=caption)


def negation_rules_bar(df: pd.DataFrame, *, title: str, caption: str, threshold: float = 0.5) -> Chart:
    """
    Expects columns: pattern, flip_rate, n
    """
    d = df.copy()
    d["flip_pct"] = d["flip_rate"].map(_pct)
    d["color"] = d["flip_rate"].apply(lambda r: COLORS["teal"] if float(r) >= threshold else COLORS["coral"])

    fig = go.Figure(
        data=[
            go.Bar(
                x=d["flip_pct"],
                y=d["pattern"],
                orientation="h",
                marker_color=d["color"],
                text=[f"{v:.1f}%" for v in d["flip_pct"]],
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        title=title,
        xaxis_title="Flip rate (%)",
        yaxis_title="Negation pattern",
        margin=dict(l=20, r=20, t=60, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[0, max(60, float(d["flip_pct"].max()) * 1.25)]),
    )
    return Chart(fig=fig, caption=caption)


def bucket_flip_rate_bar(df: pd.DataFrame, *, title: str, caption: str) -> Chart:
    """
    Expects columns: bucket, flip_rate, n
    """
    d = df.copy()
    d["flip_pct"] = d["flip_rate"].map(_pct)
    fig = go.Figure(
        data=[
            go.Bar(
                x=d["bucket"],
                y=d["flip_pct"],
                marker_color=COLORS["teal"],
                text=[f"{v:.1f}%" for v in d["flip_pct"]],
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        title=title,
        xaxis_title="Bucket",
        yaxis_title="Flip rate (%)",
        margin=dict(l=20, r=20, t=60, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(range=[0, max(5, float(d["flip_pct"].max()) * 1.25)]),
    )
    return Chart(fig=fig, caption=caption)


def minimality_distribution_bar(df: pd.DataFrame, *, title: str, caption: str) -> Chart:
    """
    Expects columns: min_drugs_to_flip, pct (0..1)
    """
    d = df.copy()
    d["pct"] = d["pct"].map(_pct)
    fig = go.Figure(
        data=[
            go.Bar(
                x=d["min_drugs_to_flip"],
                y=d["pct"],
                marker_color=COLORS["navy"],
                text=[f"{v:.1f}%" for v in d["pct"]],
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        title=title,
        xaxis_title="Minimum # of drugs to mask to flip",
        yaxis_title="Percent of sentences (%)",
        margin=dict(l=20, r=20, t=60, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(range=[0, max(5, float(d["pct"].max()) * 1.25)]),
    )
    return Chart(fig=fig, caption=caption)


def interaction_direction_pie(rates: dict[str, float], *, title: str, caption: str) -> Chart:
    coop = float(rates.get("cooperative", 0.0))
    comp = float(rates.get("competitive", 0.0))
    fig = go.Figure(
        data=[
            go.Pie(
                labels=["Cooperative (same direction)", "Competitive (opposite direction)"],
                values=[coop, comp],
                hole=0.4,
                marker_colors=[COLORS["teal"], COLORS["coral"]],
                textinfo="percent",
            )
        ]
    )
    fig.update_layout(
        title=title,
        margin=dict(l=20, r=20, t=60, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return Chart(fig=fig, caption=caption)


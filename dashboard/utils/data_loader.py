from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import streamlit as st


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_load_json(filename: str) -> Optional[Any]:
    path = RESULTS_DIR / filename
    try:
        if not path.exists():
            return None
        return _read_json(path)
    except Exception:
        return None


def missing_files(filenames: list[str]) -> list[str]:
    missing: list[str] = []
    for fn in filenames:
        if not (RESULTS_DIR / fn).exists():
            missing.append(fn)
    return missing


@dataclass(frozen=True)
class SensitivityRatio:
    point_estimate: float
    ci_95_low: float
    ci_95_high: float


@dataclass(frozen=True)
class McNemarResult:
    statistic: float
    p_value: float
    significant: bool
    n_paired: int
    contingency_table: list[list[int]]


@st.cache_data(show_spinner=False)
def load_statistical_tests() -> Optional[dict[str, Any]]:
    return safe_load_json("statistical_tests.json")


def get_sensitivity_ratio(stats: Optional[dict[str, Any]]) -> Optional[SensitivityRatio]:
    if not stats:
        return None
    sr = stats.get("sensitivity_ratio")
    if not isinstance(sr, dict):
        return None
    try:
        return SensitivityRatio(
            point_estimate=float(sr["point_estimate"]),
            ci_95_low=float(sr["ci_95_low"]),
            ci_95_high=float(sr["ci_95_high"]),
        )
    except Exception:
        return None


def get_mcnemar(stats: Optional[dict[str, Any]]) -> Optional[McNemarResult]:
    if not stats:
        return None
    mc = stats.get("mcnemar")
    if not isinstance(mc, dict):
        return None
    try:
        return McNemarResult(
            statistic=float(mc["statistic"]),
            p_value=float(mc["p_value"]),
            significant=bool(mc["significant"]),
            n_paired=int(mc["n_paired"]),
            contingency_table=mc["contingency_table"],
        )
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_family_a() -> Optional[dict[str, Any]]:
    return safe_load_json("family_a_with_confidence.json")


@st.cache_data(show_spinner=False)
def load_family_b() -> Optional[dict[str, Any]]:
    return safe_load_json("family_b_with_confidence.json")


@st.cache_data(show_spinner=False)
def load_family_c() -> Optional[dict[str, Any]]:
    return safe_load_json("family_c_with_confidence.json")


def _per_sentence_df(payload: Optional[dict[str, Any]]) -> Optional[pd.DataFrame]:
    if not payload or "per_sentence" not in payload:
        return None
    rows = payload.get("per_sentence")
    if not isinstance(rows, list):
        return None
    if len(rows) == 0:
        return pd.DataFrame()
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def load_family_a_sentences() -> Optional[pd.DataFrame]:
    return _per_sentence_df(load_family_a())


@st.cache_data(show_spinner=False)
def load_family_b_sentences() -> Optional[pd.DataFrame]:
    return _per_sentence_df(load_family_b())


@st.cache_data(show_spinner=False)
def load_family_c_sentences() -> Optional[pd.DataFrame]:
    return _per_sentence_df(load_family_c())


@st.cache_data(show_spinner=False)
def load_family_a_non_ade_counts() -> Optional[dict[str, int]]:
    flips = safe_load_json("masking_non_ade_flips.json")
    nonflips = safe_load_json("masking_non_ade_nonflips.json")
    if flips is None or nonflips is None:
        return None
    if not isinstance(flips, list) or not isinstance(nonflips, list):
        return None
    return {"flipped": len(flips), "unflipped": len(nonflips), "total": len(flips) + len(nonflips)}


@st.cache_data(show_spinner=False)
def load_error_patterns() -> Optional[dict[str, Any]]:
    return safe_load_json("error_patterns.json")


def error_patterns_frames(error_patterns: Optional[dict[str, Any]]) -> dict[str, pd.DataFrame]:
    if not error_patterns:
        return {}

    frames: dict[str, pd.DataFrame] = {}
    for key in ["by_sentence_length", "by_drug_mention_count", "by_causal_marker"]:
        section = error_patterns.get(key)
        if not isinstance(section, dict):
            continue
        rows = []
        for bucket, vals in section.items():
            if not isinstance(vals, dict):
                continue
            rows.append(
                {
                    "bucket": str(bucket),
                    "flip_rate": float(vals.get("flip_rate", 0.0)),
                    "n": int(vals.get("n", 0)),
                    "flipped": int(vals.get("flipped", 0)),
                }
            )
        frames[key] = pd.DataFrame(rows)
    return frames


def paired_failure_modes(error_patterns: Optional[dict[str, Any]]) -> Optional[dict[str, int]]:
    if not error_patterns:
        return None
    pfm = error_patterns.get("paired_failure_modes")
    if not isinstance(pfm, dict):
        return None
    keys = ["both_flipped", "only_a", "only_c", "neither", "total_paired"]
    try:
        return {k: int(pfm[k]) for k in keys if k in pfm}
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_negation_analysis() -> Optional[dict[str, Any]]:
    return safe_load_json("negation_analysis.json")


def negation_rule_frame(neg: Optional[dict[str, Any]]) -> Optional[pd.DataFrame]:
    if not neg:
        return None
    by_rule = neg.get("by_rule")
    if not isinstance(by_rule, dict):
        return None
    rows = []
    for rule, vals in by_rule.items():
        if not isinstance(vals, dict):
            continue
        label = str(rule)
        label = label.replace("\\b", "").strip()
        label = label.strip("()")
        rows.append({"pattern": label, "flip_rate": float(vals.get("flip_rate", 0.0)), "n": int(vals.get("n", 0))})
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df
    return df.sort_values("flip_rate", ascending=True, ignore_index=True)


def negation_per_sentence_frame(neg: Optional[dict[str, Any]]) -> Optional[pd.DataFrame]:
    if not neg:
        return None
    rows = neg.get("per_sentence")
    if not isinstance(rows, list):
        return None
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def load_minimality() -> Optional[dict[str, Any]]:
    return safe_load_json("minimality_analysis.json")


def minimality_distribution(min_payload: Optional[dict[str, Any]]) -> Optional[pd.DataFrame]:
    if not min_payload:
        return None
    agg = min_payload.get("aggregate")
    if not isinstance(agg, dict):
        return None
    dist = agg.get("min_drug_distribution")
    if not isinstance(dist, dict):
        return None
    rows = []
    total = sum(int(v) for v in dist.values()) if dist else 0
    for k, v in dist.items():
        rows.append({"min_drugs_to_flip": str(k), "count": int(v), "pct": (int(v) / total) if total else 0.0})
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df
    # order numeric-ish keys, keep '5' etc.
    def _key(x: str) -> int:
        try:
            return int(x)
        except Exception:
            return 999

    return df.sort_values("min_drugs_to_flip", key=lambda s: s.map(_key), ignore_index=True)


@st.cache_data(show_spinner=False)
def load_shap_interactions() -> Optional[dict[str, Any]]:
    return safe_load_json("shap_interactions.json")


def shap_interaction_direction_rates(shap_interactions: Optional[dict[str, Any]]) -> Optional[dict[str, float]]:
    if not shap_interactions:
        return None
    agg = shap_interactions.get("aggregate")
    if not isinstance(agg, dict):
        return None
    same = agg.get("mean_same_direction_rate")
    if same is None:
        return None
    same = float(same)
    return {"cooperative": same, "competitive": max(0.0, 1.0 - same)}


@st.cache_data(show_spinner=False)
def load_attention_summary() -> Optional[dict[str, Any]]:
    return safe_load_json("attention_summary.json")


@st.cache_data(show_spinner=False)
def load_shap_full() -> Optional[dict[str, Any]]:
    # In this dashboard, `shap_analysis_full.json` is copied into dashboard/results/
    return safe_load_json("shap_analysis_full.json")


@st.cache_data(show_spinner=False)
def load_lime_full() -> Optional[dict[str, Any]]:
    return safe_load_json("lime_analysis_full.json")


@st.cache_data(show_spinner=False)
def load_ig_full() -> Optional[dict[str, Any]]:
    return safe_load_json("integrated_gradients_full.json")


def xai_aggregate_row(payload: Optional[dict[str, Any]], method: str) -> Optional[dict[str, float]]:
    if not payload or not isinstance(payload, dict):
        return None
    agg = payload.get("aggregate")
    if not isinstance(agg, dict):
        return None

    ratio_key_candidates = ["drug_nondrug_ratio"]
    pct_key_candidates = [
        "pct_drug_in_top3",
        "pct_sentences_drug_in_top3",
    ]

    ratio = None
    for k in ratio_key_candidates:
        if k in agg:
            ratio = float(agg[k])
            break
    pct = None
    for k in pct_key_candidates:
        if k in agg:
            pct = float(agg[k])
            break

    if ratio is None and pct is None:
        return None
    return {
        "method": float("nan"),  # caller may overwrite or ignore
        "ratio": float(ratio) if ratio is not None else float("nan"),
        "pct_top3": float(pct) if pct is not None else float("nan"),
        "_method_name": method,
    }


@st.cache_data(show_spinner=False)
def load_drug_vocab() -> Optional[list[str]]:
    payload = safe_load_json("extracted_drug_names.json")
    if payload is None:
        return None
    if isinstance(payload, list):
        return [str(x) for x in payload if isinstance(x, (str, int, float))]
    return None


from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import streamlit as st

from utils import data_loader


def _mask_drugs(text: str, drug_vocab: list[str]) -> str:
    # Match your notebook's behavior: replace each matched span with a single [MASK].
    # Use longest-first to avoid substring shadowing.
    sorted_terms = sorted({t.strip() for t in drug_vocab if t and len(t.strip()) > 1}, key=len, reverse=True)
    out = text
    for term in sorted_terms[:5000]:
        # keep this reasonably fast; vocab is large
        pattern = r"\\b" + re.escape(term) + r"\\b"
        out = re.sub(pattern, "[MASK]", out, flags=re.IGNORECASE)
    return out


@st.cache_resource(show_spinner=False)
def _load_model(model_path: str):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch

    p = Path(model_path)
    tokenizer = AutoTokenizer.from_pretrained(str(p))
    model = AutoModelForSequenceClassification.from_pretrained(str(p))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device


def _predict_proba(text: str, tokenizer, model, device) -> float:
    import torch

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()[0]
    # class 1 is ADE in this project
    return float(probs[1])


st.title("Live Demo (optional)")
st.markdown(
    "This page optionally loads a local fine-tuned BioBERT checkpoint and runs a live prediction. "
    "If no model is available, it falls back to a static example from the dataset."
)

model_path = st.text_input("Local model path (directory)", value="")
text = st.text_area("Clinical sentence", value="Intravenous azithromycin-induced ototoxicity.", height=120)

drug_vocab = data_loader.load_drug_vocab() or []

if model_path and Path(model_path).exists():
    try:
        tokenizer, model, device = _load_model(model_path)
        p_orig = _predict_proba(text, tokenizer, model, device)
        masked = _mask_drugs(text, drug_vocab) if drug_vocab else text
        p_mask = _predict_proba(masked, tokenizer, model, device)

        st.subheader("Prediction")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("P(ADE) original", f"{p_orig:.4f}")
            st.write("**Label:**", "ADE" if p_orig >= 0.5 else "non-ADE")
        with c2:
            st.metric("P(ADE) masked", f"{p_mask:.4f}")
            st.write("**Label:**", "ADE" if p_mask >= 0.5 else "non-ADE")

        st.subheader("Masked sentence")
        st.write(masked)
        st.caption(
            "Masking is done using the extracted drug vocabulary and replaces each matched drug span with a single `[MASK]`, "
            "matching the project’s counterfactual masking convention."
        )

        flipped = (p_orig >= 0.5) != (p_mask >= 0.5)
        st.success("Prediction flipped after masking." if flipped else "Prediction did not flip after masking.")
    except Exception as e:
        st.warning(f"Could not load or run the model at that path. Falling back to a static example. Error: {e}")
        model_path = ""

if not model_path:
    st.subheader("Static example (no model loaded)")
    fa = data_loader.load_family_a_sentences()
    if fa is None or len(fa) == 0:
        st.info("No family data available to show an example.")
    else:
        r = fa.iloc[0]
        st.write("**Original**:", r.get("original_text", "—"))
        st.write("**Masked**:", r.get("masked_text", "—"))
        st.metric("P(ADE) original", f"{float(r.get('original_prob', 0.0)):.4f}")
        st.metric("P(ADE) masked", f"{float(r.get('masked_prob', 0.0)):.4f}")
        st.write("**Flipped**:", bool(r.get("flipped", False)))
        st.caption(
            "To enable live inference, provide a local HuggingFace checkpoint directory (tokenizer + model weights) in the text box above."
        )


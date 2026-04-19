# Extending ADE detection with Explainable Counterfactuals and BioBERT

This project demonstrates an end-to-end pipeline for Adverse Drug Event (ADE) detection from clinical text, integrating a fine-tuned BioBERT model with counterfactual explanations to achieve both high performance and crucial interpretability for clinical AI.

## Abstract

Adverse Drug Events (ADEs) pose a significant clinical safety concern. While NLP models like BioBERT excel at ADE classification, their "black box" nature can hinder real-world adoption. [cite: 2] This work bridges this gap by combining BioBERT fine-tuning with counterfactual analysis. [cite: 3] We achieve high classification performance (F1: 95.6%) on the ADE-Corpus V2 and reveal model sensitivities to drug name masking (76.5% flip rate in ADE sentences vs. 1.9% in non-ADE) and brand name swapping (8.6% overall flip rate), paving the way for transparent and trustworthy clinical AI. [cite: 4, 5]

## Motivation

Automated ADE detection is vital for reducing manual effort, strengthening drug safety, and ensuring prompt patient care. [cite: 6, 7] However, clinical adoption of powerful deep learning models is often limited by their lack of transparency. Clinicians require understandable reasons behind AI predictions to justify decisions and build trust. [cite: 9] This project addresses this need by focusing on both high performance and explainability in ADE detection. [cite: 10]

## Methodology

1.  **Model:** We fine-tuned the `dmis-lab/biobert-base-cased-v1.1` model. [cite: 33]
2.  **Dataset:** The ADE-Corpus V2 (specifically `Ade_corpus_v2_classification` for classification and `Ade_corpus_v2_drug_ade_relation` for entity identification) was used, comprising over 23,000 clinical sentences. [cite: 23, 24, 29]
3.  **Training:** The model was trained for 10 epochs, batch size 16, max sequence length 128 tokens using Hugging Face Transformers. [cite: 34]
4.  **Counterfactual Analysis:**
    * **Drug Name Masking:** Drug mentions in 6,821 ADE-positive and 5,093 non-ADE sentences were replaced with `[MASK]` tokens to measure prediction flip rates. [cite: 26, 27]
    * **Brand Name Substitution:** Generic drug names (1,049 unique, 541 mapped to brand names via RxNorm API) were replaced with brand names in 8,448 sentences. [cite: 31, 32, 39]

## Key Results & Findings

* **Classification Performance:**
    * Accuracy: 96.3% [cite: 42, 43]
    * Precision (macro): 95.3% [cite: 42, 43]
    * Recall (macro): 96.2% [cite: 42, 43]
    * **F1-score (macro): 95.6%** [cite: 42, 43]
* **Drug Masking Impact:**
    * **76.5%** of ADE-positive sentences had predictions flipped when drug names were masked (5,217 out of 6,821). [cite: 45, 49]
    * Only **1.9%** of non-ADE sentences flipped (94 out of 5,093). [cite: 46, 49]
    * This highlights a strong reliance on the drug token itself for positive ADE classification. [cite: 47]
* **Brand Name Substitution Impact:**
    * **8.6%** overall prediction flip rate when generic names were swapped with brand names (726 out of 8,448 sentences). [cite: 54, 49]
    * **13.1%** flip rate for ADE-positive sentences within this subset (675 out of 5,155). [cite: 55, 49]
    * This indicates some lexical inflexibility, suggesting the model is less robust when encountering less frequent brand names. [cite: 66, 67]

## Discussion & Limitations

The high flip rate from drug masking in ADE sentences (76.5%) confirms drug names are powerful predictors. [cite: 61] The lower, yet significant, flip rate from brand substitution (13.1% for ADE sentences) suggests a need for incorporating more comprehensive drug synonym knowledge during training. [cite: 66, 68]

Limitations include reliance on a single model (BioBERT) and dataset (ADE-Corpus V2), incomplete brand name mapping (approx. 51.6% yield), a limited set of counterfactual types, and using flip rate as the primary metric for change. [cite: 71, 72, 73, 74, 75, 76, 77, 78]

## Conclusion & Impact

This work presents an effective approach for ADE detection that integrates high-performing BioBERT with explainability through counterfactual analysis. [cite: 79, 84] By systematically perturbing inputs, we made the model's decision boundaries and reliance on drug mentions more transparent. [cite: 85] These findings support the development of safer, more interpretable AI applications suitable for clinical environments where performance, transparency, and trust are paramount. [cite: 86, 88, 90]

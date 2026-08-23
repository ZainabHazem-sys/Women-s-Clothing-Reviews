"""
Resource loading (Hugging Face) and shared inference helpers.

Model, vectorizer, and dataset all live on the Hugging Face repo below and are
downloaded once per session and cached -- nothing is retrained or refit here,
and nothing needs to be committed to this repo/GitHub.
"""

import io

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st

from nlp_pipeline import CLASS_ORDER, clean_text, preprocess_review, rating_to_sentiment, tokenize_and_filter

HF_REPO = "FatmaEissa1/Sentiment_Analysis_Womens_Clothing"
MODEL_URL = f"https://huggingface.co/{HF_REPO}/resolve/main/best_sentiment_model.pkl"
VECTORIZER_URL = f"https://huggingface.co/{HF_REPO}/resolve/main/tfidf_vectorizer.pkl"
DATASET_URL = f"https://huggingface.co/{HF_REPO}/resolve/main/Womens%20Clothing%20E-Commerce%20Reviews.csv"


def _download_bytes(url: str, timeout: int = 60) -> bytes:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.content


@st.cache_resource(show_spinner="Loading sentiment model from Hugging Face...")
def load_model():
    raw = _download_bytes(MODEL_URL)
    return joblib.load(io.BytesIO(raw))


@st.cache_resource(show_spinner="Loading TF-IDF vectorizer from Hugging Face...")
def load_vectorizer():
    raw = _download_bytes(VECTORIZER_URL)
    return joblib.load(io.BytesIO(raw))


@st.cache_data(show_spinner="Loading dataset from Hugging Face...")
def load_dataset() -> pd.DataFrame:
    raw = _download_bytes(DATASET_URL)
    df = pd.read_csv(io.BytesIO(raw))

    # Mirrors notebook Section 4: drop leftover index column / Title (not the
    # NLP feature), and drop rows with no Review Text (essential feature).
    drop_cols = [c for c in ["Unnamed: 0", "Title"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    df = df.dropna(subset=["Review Text"]).reset_index(drop=True)

    # Mirrors notebook Section 5: build the 3-class target from Rating.
    df["Sentiment"] = df["Rating"].apply(rating_to_sentiment)

    # Mirrors notebook Sections 6-7: same cleaning / tokenization pipeline
    # used to produce the text the model was trained on.
    df["Clean_Review"] = df["Review Text"].apply(clean_text)
    df["Tokens"] = df["Clean_Review"].apply(tokenize_and_filter)
    df["Processed_Review"] = df["Tokens"].apply(lambda toks: " ".join(toks))
    df["Review_Length"] = df["Tokens"].apply(len)

    return df


# ---------------------------------------------------------------------------
# Inference helpers (transform + predict ONLY -- never fit / fit_transform)
# ---------------------------------------------------------------------------
def predict_single_review(raw_text: str, vectorizer, model):
    """Run one raw review through the exact notebook pipeline and the
    existing fitted vectorizer + model. Returns (label, proba_dict_or_None,
    proba_kind) where proba_kind is 'predict_proba', 'decision_function', or
    None depending on what the loaded model supports."""
    processed = preprocess_review(raw_text)
    X = vectorizer.transform([processed])  # transform only, never fit

    label = model.predict(X)[0]

    proba_dict = None
    proba_kind = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        proba_dict = dict(zip(model.classes_, proba))
        proba_kind = "predict_proba"
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X)[0]
        scores = np.atleast_1d(scores)
        if scores.shape[0] == len(model.classes_):
            exp_scores = np.exp(scores - np.max(scores))
            softmax = exp_scores / exp_scores.sum()
            proba_dict = dict(zip(model.classes_, softmax))
            proba_kind = "decision_function"

    return label, proba_dict, proba_kind


def predict_batch(processed_texts, vectorizer, model):
    """Vectorized transform + predict for many already-preprocessed review
    strings (used by Product Recommendation and PDF Analysis)."""
    X = vectorizer.transform(processed_texts)  # transform only, never fit
    return model.predict(X)


def analyze_product(clothing_id, df: pd.DataFrame, vectorizer, model):
    """Exact port of the notebook's `analyze_product` rule-based system
    (Section 22). Returns None if the Clothing ID has no reviews."""
    product_df = df[df["Clothing ID"] == clothing_id]
    if product_df.empty:
        return None

    n_reviews = len(product_df)
    avg_rating = product_df["Rating"].mean()

    rec_rate = product_df["Recommended IND"].mean() * 100
    not_rec_rate = 100 - rec_rate

    preds = predict_batch(product_df["Processed_Review"], vectorizer, model)
    sentiment_counts = pd.Series(preds).value_counts()
    sentiment_pct = (sentiment_counts / n_reviews * 100).reindex(CLASS_ORDER).fillna(0)

    positive_pct = sentiment_pct["Positive"]
    neutral_pct = sentiment_pct["Neutral"]
    negative_pct = sentiment_pct["Negative"]

    # Exact rule from the notebook -- never change independently of it.
    is_recommended = (avg_rating >= 4) and (rec_rate >= 70) and (positive_pct > negative_pct)
    final_label = "Recommended" if is_recommended else "Not Recommended"

    return {
        "clothing_id": clothing_id,
        "n_reviews": n_reviews,
        "avg_rating": avg_rating,
        "rec_rate": rec_rate,
        "not_rec_rate": not_rec_rate,
        "positive_pct": positive_pct,
        "neutral_pct": neutral_pct,
        "negative_pct": negative_pct,
        "final_label": final_label,
        "product_df": product_df,
    }

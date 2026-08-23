"""
Text preprocessing pipeline.

This module is an EXACT mirror of the cleaning / tokenization pipeline used in
the source notebook (Sections 6-7) to train `best_sentiment_model.pkl` and fit
`tfidf_vectorizer.pkl`. It must never be changed independently of the notebook,
since the fitted vectorizer's vocabulary was built on text produced by this
exact pipeline. All inference pages (Review Prediction, Product Recommendation,
PDF Analysis) import from here so every prediction path stays consistent.
"""

import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# ---------------------------------------------------------------------------
# Sentiment label definition (Notebook Section 5)
# ---------------------------------------------------------------------------
CLASS_ORDER = ["Negative", "Neutral", "Positive"]


def rating_to_sentiment(rating):
    """Rating -> 3-class sentiment label. Used only to build ground-truth
    labels from historical data (e.g. for dashboard charts) -- never used as
    a model feature."""
    if rating <= 2:
        return "Negative"
    elif rating == 3:
        return "Neutral"
    else:
        return "Positive"


# ---------------------------------------------------------------------------
# Text cleaning (Notebook Section 6)
# ---------------------------------------------------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)   # remove URLs
    text = re.sub(r"[^a-z\s']", " ", text)          # keep letters, spaces, apostrophes
    text = re.sub(r"\s+", " ", text).strip()        # normalize whitespace
    return text


# ---------------------------------------------------------------------------
# Tokenization & stopword removal (Notebook Section 7)
# ---------------------------------------------------------------------------
NEGATION_WORDS = {"not", "no", "never", "without", "nothing", "nor", "cannot", "n't"}
STOP_WORDS = ENGLISH_STOP_WORDS - NEGATION_WORDS


def tokenize_and_filter(text):
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
    return tokens


def preprocess_review(raw_text):
    """Full pipeline for a single raw review string: clean -> tokenize/filter
    -> re-join into the exact 'Processed_Review' format the TF-IDF vectorizer
    was fit on. This is the ONLY function inference pages should call before
    handing text to `tfidf.transform(...)`."""
    cleaned = clean_text(raw_text)
    tokens = tokenize_and_filter(cleaned)
    return " ".join(tokens)

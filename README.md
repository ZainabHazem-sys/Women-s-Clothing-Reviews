# Clothing Review System

A system for analyzing customer reviews of women's clothing e-commerce products, built on top of a pre-trained sentiment analysis model (three classes: Negative / Neutral / Positive), a rule-based product recommendation system, and batch PDF review analysis.

The application is built entirely with **Streamlit** and performs **no training or retraining** at runtime — the model, TF-IDF vectorizer, and dataset are all loaded ready-made from Hugging Face.

---

## Table of Contents

- [Overview](#overview)
- [Pages](#pages)
- [Text Preprocessing Pipeline](#text-preprocessing-pipeline)
- [Product Recommendation System](#product-recommendation-system)
- [Resources Used (Hugging Face)](#resources-used-hugging-face)
- [Project Structure](#project-structure)
- [Running Locally](#running-locally)
- [Deploying to Streamlit Cloud](#deploying-to-streamlit-cloud)
- [Important Technical Notes](#important-technical-notes)

---

## Overview

The project is framed as a three-class classification problem:

| Rating | Sentiment |
|---|---|
| 1–2 | Negative |
| 3 | Neutral |
| 4–5 | Positive |

The model was pre-trained and selected based on the best **Macro F1 Score** among several classifiers (Logistic Regression, Linear SVM, Naive Bayes, SGD, Random Forest, Extra Trees, Decision Tree), then saved as `best_sentiment_model.pkl` along with its fitted vectorizer `tfidf_vectorizer.pkl`.

This application **does not retrain any model** — it only calls `transform()` and `predict()` on the existing, already-fitted resources.

---

## Pages

The application has four main pages, navigated through a top navigation bar:

### 1. Dataset Dashboard
A full interactive dashboard for the original dataset, including:
- Key performance indicators (KPIs): total reviews, number of products, average rating, recommendation rate, average age, average positive feedback count.
- Interactive filters: Department, Class, Rating, Age range, Recommendation status.
- Plotly visualizations: rating distribution, recommendation split, reviews by department/class, age distribution, sentiment distribution, average rating by department, sentiment by department, and recommendation rate by department.

### 2. Review Prediction
An interface for analyzing a single new review:
- Enter the review text.
- It's passed through the exact same preprocessing pipeline used during training.
- The final sentiment (Negative / Neutral / Positive) is displayed, along with a confidence score when the loaded model supports it.

### 3. Product Recommendation
Search for a product by `Clothing ID` and view:
- Number of reviews, average rating, historical recommendation rate.
- Predicted sentiment mix (Positive / Neutral / Negative) by running the existing model on the product's reviews.
- A clear final verdict: **Recommended** or **Not Recommended**, based on a transparent rule rather than a separately trained model.

### 4. PDF Analysis
Batch analysis of a PDF containing multiple reviews:
- Extract text from the file.
- Split it into individual reviews.
- Classify each review using the existing model (no retraining).
- Display the sentiment distribution, a filterable results table, and a CSV export of the results.

---

## Text Preprocessing Pipeline

The exact same preprocessing steps used to train the model, unchanged:

1. **Text Cleaning**: lowercase the text, remove URLs, keep only letters, spaces, and apostrophes.
2. **Tokenization & Stopword Removal**: remove standard English stopwords, **except negation words** (not, no, never, without...), since they carry important sentiment meaning.
3. **Processed Review**: rejoin the remaining tokens into a single string ready to be passed to the TF-IDF vectorizer.

These steps live in a single file (`nlp_pipeline.py`) and are imported by every page, so all predictions stay perfectly consistent with the trained model.

---

## Product Recommendation System

A transparent, rule-based system — not a separately trained model:

```
A product is Recommended if all of the following hold:

1) Average Rating >= 4
2) Historical Recommendation Rate >= 70%
3) Positive Sentiment % > Negative Sentiment %

Otherwise: Not Recommended
```

---

## Resources Used (Hugging Face)

The following resources are loaded directly from Hugging Face at runtime (no need to commit them to GitHub):

| Resource | Description |
|---|---|
| `best_sentiment_model.pkl` | The trained model (best model by Macro F1) |
| `tfidf_vectorizer.pkl` | The fitted vectorizer (used only via `transform`) |
| `Womens Clothing E-Commerce Reviews.csv` | The original dataset |

Caching:
- `@st.cache_resource` for loading the model and vectorizer (loaded once per session).
- `@st.cache_data` for loading and preprocessing the dataset.

---

## Project Structure

```
clothing-review-system/
├── app.py              # Main entry point + top navigation + all four pages
├── data_loader.py       # Hugging Face resource loading + inference helpers
├── nlp_pipeline.py       # Text preprocessing (single source of truth for all pages)
├── pdf_utils.py          # PDF text extraction and review splitting
├── styles.py             # Shared theme (colors, cards, top navigation)
└── requirements.txt      # Python dependencies
```

---

## Running Locally

```bash
# 1) Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate    # on Windows: venv\Scripts\activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Run the app
streamlit run app.py
```

The app will automatically download the model, vectorizer, and dataset from Hugging Face on first run.

---

## Deploying to Streamlit Cloud

1. Push the files (`app.py`, `data_loader.py`, `nlp_pipeline.py`, `pdf_utils.py`, `styles.py`, `requirements.txt`) to a GitHub repo.
2. Connect the repo to [Streamlit Cloud](https://streamlit.io/cloud).
3. Set `app.py` as the main entry point.
4. No need to upload the model or dataset — they are downloaded automatically from Hugging Face at runtime.

---

## Important Technical Notes

- **No training happens inside the app** — only `transform()` and `predict()` are called on the existing fitted resources.
- If the saved model doesn't support `predict_proba()` (e.g. some SVM configurations), the app shows an approximate confidence derived from `decision_function`, or just the predicted label with no confidence score.
- Extracting reviews from PDF files uses a reasonable strategy (numbered items, paragraph breaks, then individual lines) without inventing any text that isn't actually in the file.
- The design is consistent across all pages, using a soft, light color palette (blush / off-white / charcoal).

"""PDF text extraction + heuristic splitting into individual reviews.

The model never sees the PDF directly -- text is extracted, split into
review-sized chunks, run through the exact same `preprocess_review` pipeline
used everywhere else, then transformed/predicted with the existing fitted
vectorizer + model (transform/predict only, no fitting)."""

import re

import pdfplumber


def extract_text_from_pdf(file_bytes) -> str:
    pages_text = []
    with pdfplumber.open(file_bytes) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            pages_text.append(text)
    return "\n".join(pages_text)


_NUMBERED_PATTERN = re.compile(r"\n\s*\d{1,4}[\.\)]\s+")


def split_into_reviews(full_text: str, min_words: int = 3):
    """Reasonable, non-inventive extraction strategy:
    1) If the text contains a clear numbered-list pattern (e.g. '1.', '23)'),
       split on that -- this is the most common structure for a PDF of many
       reviews.
    2) Otherwise split on blank-line paragraph breaks.
    3) Otherwise fall back to one review per non-trivial line.
    In every case, fragments are only whitespace-trimmed -- no text is added,
    reworded, or guessed."""
    full_text = full_text.replace("\r\n", "\n")

    numbered_hits = _NUMBERED_PATTERN.findall("\n" + full_text)
    if len(numbered_hits) >= 2:
        chunks = _NUMBERED_PATTERN.split("\n" + full_text)
        chunks = [c.strip() for c in chunks]
    else:
        paragraph_chunks = [p.strip() for p in re.split(r"\n\s*\n", full_text)]
        if len([p for p in paragraph_chunks if p]) >= 2:
            chunks = paragraph_chunks
        else:
            chunks = [ln.strip() for ln in full_text.split("\n")]

    reviews = []
    for chunk in chunks:
        chunk = re.sub(r"\s+", " ", chunk).strip()
        if not chunk:
            continue
        if len(chunk.split()) < min_words:
            continue
        reviews.append(chunk)

    return reviews

"""
intent_router.py

Routes a customer message to a support category using simple keyword
matching combined with TF-IDF cosine similarity against category exemplars.
Designed to be lightweight enough for a 48-hour hackathon build, with a
clear extension path to a trained intent classifier later.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CATEGORY_EXEMPLARS = {
    "Billing": [
        "I was charged twice for my order",
        "refund my payment please",
        "my invoice looks wrong",
        "subscription billing issue",
        "credit card charge dispute",
    ],
    "Technical": [
        "the app keeps crashing",
        "I can't log into my account",
        "the website is showing an error",
        "feature is not working properly",
        "bug in the system",
    ],
    "Shipping": [
        "where is my order",
        "package has not arrived yet",
        "delivery is delayed",
        "tracking number not working",
        "wrong item was delivered",
    ],
    "General": [
        "I have a question about your product",
        "how do I use this feature",
        "general feedback about my experience",
        "I want to know more about pricing",
    ],
}

_LOW_CONFIDENCE_THRESHOLD = 0.18

_categories = list(CATEGORY_EXEMPLARS.keys())
_exemplar_texts = []
_exemplar_labels = []
for category, examples in CATEGORY_EXEMPLARS.items():
    for ex in examples:
        _exemplar_texts.append(ex)
        _exemplar_labels.append(category)

_vectorizer = TfidfVectorizer()
_exemplar_matrix = _vectorizer.fit_transform(_exemplar_texts)


def route_category(cleaned_text):
    """
    Returns a dict: {"category": ..., "confidence": ..., "is_ambiguous": bool}
    """
    if not cleaned_text.strip():
        return {"category": "General", "confidence": 0.0, "is_ambiguous": True}

    query_vec = _vectorizer.transform([cleaned_text])
    similarities = cosine_similarity(query_vec, _exemplar_matrix)[0]

    best_idx = similarities.argmax()
    best_score = float(similarities[best_idx])
    best_category = _exemplar_labels[best_idx]

    is_ambiguous = best_score < _LOW_CONFIDENCE_THRESHOLD

    return {
        "category": best_category if not is_ambiguous else "Unclear",
        "confidence": round(best_score, 2),
        "is_ambiguous": is_ambiguous,
    }

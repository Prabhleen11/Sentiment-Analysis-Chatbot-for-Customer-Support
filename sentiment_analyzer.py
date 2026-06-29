"""
sentiment_analyzer.py

Primary sentiment scoring via VADER (well-suited to short, informal text like
chat/support messages). Falls back to a trained TF-IDF + Logistic Regression
classifier if a backup model file is available and VADER's confidence is low.
"""

import os
import joblib

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

try:
    _vader = SentimentIntensityAnalyzer()
except LookupError:
    nltk.download("vader_lexicon")
    _vader = SentimentIntensityAnalyzer()

BACKUP_MODEL_PATH = os.path.join("models", "sentiment_backup_model.joblib")
BACKUP_VECTORIZER_PATH = os.path.join("models", "sentiment_vectorizer.joblib")

_backup_model = None
_backup_vectorizer = None
if os.path.exists(BACKUP_MODEL_PATH) and os.path.exists(BACKUP_VECTORIZER_PATH):
    _backup_model = joblib.load(BACKUP_MODEL_PATH)
    _backup_vectorizer = joblib.load(BACKUP_VECTORIZER_PATH)


def _label_from_compound(compound_score):
    if compound_score >= 0.05:
        return "Positive"
    elif compound_score <= -0.05:
        return "Negative"
    return "Neutral"


def analyze_sentiment(cleaned_text, low_confidence_threshold=0.15):
    """
    Returns a dict: {"label": ..., "confidence": ..., "source": "vader" | "backup_model"}
    """
    scores = _vader.polarity_scores(cleaned_text)
    compound = scores["compound"]
    label = _label_from_compound(compound)
    confidence = abs(compound)

    # If VADER is unsure and a trained backup model is available, defer to it.
    if confidence < low_confidence_threshold and _backup_model is not None:
        X = _backup_vectorizer.transform([cleaned_text])
        pred_label = _backup_model.predict(X)[0]
        pred_proba = max(_backup_model.predict_proba(X)[0])
        return {"label": pred_label, "confidence": round(float(pred_proba), 2), "source": "backup_model"}

    return {"label": label, "confidence": round(float(confidence), 2), "source": "vader"}

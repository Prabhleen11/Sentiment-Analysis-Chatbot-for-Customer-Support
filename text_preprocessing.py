"""
text_preprocessing.py

Cleans and normalizes raw customer support text before sentiment
analysis and intent routing.
"""

import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

_lemmatizer = WordNetLemmatizer()

try:
    _STOPWORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    _STOPWORDS = set(stopwords.words("english"))

# Keep negation words — they matter a lot for sentiment ("not happy" != "happy")
_NEGATIONS = {"no", "not", "nor", "never", "n't"}
_STOPWORDS = _STOPWORDS - _NEGATIONS


def clean_text(text):
    """Lowercases, strips URLs/punctuation/extra whitespace."""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_and_lemmatize(text, remove_stopwords=True):
    """Tokenizes cleaned text and lemmatizes each token."""
    try:
        tokens = word_tokenize(text)
    except LookupError:
        nltk.download("punkt")
        tokens = word_tokenize(text)

    if remove_stopwords:
        tokens = [t for t in tokens if t not in _STOPWORDS]

    tokens = [_lemmatizer.lemmatize(t) for t in tokens if t not in string.punctuation]
    return tokens


def preprocess(text, remove_stopwords=True):
    """
    Full preprocessing pipeline: clean -> tokenize -> lemmatize.
    Returns both the cleaned string (for VADER, which handles raw-ish text well)
    and the lemmatized token list (for the TF-IDF backup classifier).
    """
    cleaned = clean_text(text)
    tokens = tokenize_and_lemmatize(cleaned, remove_stopwords=remove_stopwords)
    return cleaned, tokens

"""
train_backup_classifier.py

Trains a TF-IDF + Logistic Regression sentiment classifier as a fallback for
cases where VADER's confidence is low. Expects a CSV with columns:
    text, label   (label in {"Positive", "Neutral", "Negative"})

Usage:
    python train_backup_classifier.py --data_path data/sample_messages.csv
"""

import argparse
import os

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from text_preprocessing import clean_text


def parse_args():
    parser = argparse.ArgumentParser(description="Train backup sentiment classifier")
    parser.add_argument("--data_path", type=str, required=True, help="CSV with 'text' and 'label' columns")
    parser.add_argument("--output_dir", type=str, default="models", help="Where to save model artifacts")
    parser.add_argument("--test_size", type=float, default=0.2)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.data_path)
    df["cleaned"] = df["text"].apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df["cleaned"], df["label"], test_size=args.test_size, random_state=42, stratify=df["label"]
    )

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train_vec, y_train)

    preds = model.predict(X_test_vec)
    print(classification_report(y_test, preds))

    joblib.dump(model, os.path.join(args.output_dir, "sentiment_backup_model.joblib"))
    joblib.dump(vectorizer, os.path.join(args.output_dir, "sentiment_vectorizer.joblib"))
    print(f"Saved backup model and vectorizer to {args.output_dir}/")


if __name__ == "__main__":
    main()

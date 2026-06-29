[README.md](https://github.com/user-attachments/files/29454061/README.md)
# Sentiment-Analysis-Chatbot-for-Customer-Support
An NLP-powered chatbot that analyzes customer sentiment from support messages and routes queries to the right category (Billing, Technical, Shipping). Uses VADER sentiment analysis, TF-IDF-based intent routing, and a Flask API, with empathetic, escalation-flagged replies for negative-sentiment messages.
# Sentiment Analysis Chatbot for Customer Support

Built during **Techsprint – 48 Hour Hackathon, GRAFEST 2025**.

## Overview

A rule-assisted, NLP-powered chatbot that analyzes the sentiment of incoming
customer support messages and routes them to the right support category
(e.g. Billing, Technical, Shipping, General Feedback). The goal was to help
support teams triage incoming queries faster by automatically flagging
frustrated or urgent customers and surfacing relevant auto-replies before a
human agent steps in.

## Problem Statement

Support teams often face a flood of incoming messages with no easy way to
prioritize which ones need urgent human attention. A negative-sentiment
message about a billing issue should be treated very differently from a
neutral question about shipping times. This project explores whether a
lightweight sentiment + intent pipeline can meaningfully improve triage
speed and reply relevance, even without a large labeled dataset.

## Approach

1. **Text preprocessing** — Incoming messages are cleaned (lowercasing,
   punctuation/stopword removal, tokenization, lemmatization) using NLTK.
2. **Sentiment classification** — A pretrained sentiment pipeline (VADER for
   short-form text, with a Scikit-learn TF-IDF + Logistic Regression model as
   an alternative/backup classifier) scores each message as
   Positive / Neutral / Negative.
3. **Intent / category routing** — A simple keyword + TF-IDF similarity
   matcher routes the message to a support category (Billing, Technical,
   Shipping, General).
4. **Response logic** — Combines sentiment + category to select a response
   template. Negative-sentiment messages are escalated with empathetic
   language and flagged for priority human follow-up; ambiguous messages
   (low confidence on both sentiment and category) fall back to a
   clarifying question instead of guessing.
5. **Serving** — A Flask app exposes a simple `/chat` endpoint and a minimal
   web form for live testing.

## Tech Stack

- **Language:** Python
- **NLP:** NLTK, VADER Sentiment
- **ML:** Scikit-learn (TF-IDF + Logistic Regression backup classifier)
- **Web Framework:** Flask
- **Version Control:** GitHub

## Project Structure

```
sentiment-chatbot/
├── README.md
├── requirements.txt
├── text_preprocessing.py     # Cleaning, tokenization, lemmatization
├── sentiment_analyzer.py     # VADER + backup ML sentiment classifier
├── intent_router.py          # Category routing logic
├── response_engine.py        # Combines sentiment + category -> reply
├── app.py                    # Flask app exposing /chat endpoint
├── train_backup_classifier.py # Optional: trains the TF-IDF + LogReg fallback model
└── templates/
    └── index.html            # Minimal chat UI for live testing
```

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download required NLTK data (one-time)
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# 3. (Optional) Train the backup TF-IDF + Logistic Regression classifier
python train_backup_classifier.py --data_path data/sample_messages.csv

# 4. Run the Flask app
python app.py
# Visit http://127.0.0.1:5000 in your browser
```

## Example

```
Input:  "I've been charged twice this month and no one is responding to my emails!"
Sentiment: Negative (confidence: 0.87)
Category:  Billing
Reply:    "I'm really sorry about the duplicate charge and the delay in hearing
           back from us. I've flagged this as urgent and routed it to our
           Billing team — someone will reach out within the hour."
```

## Results

- Correctly classified sentiment on common support phrasing patterns during
  live demo testing at the hackathon.
- Ambiguous-query fallback (asking a clarifying question instead of a wrong
  category guess) noticeably improved perceived reply relevance during
  judge demos compared to a naive keyword-only router.

## Future Improvements

- Replace the keyword/TF-IDF router with a fine-tuned intent classification
  transformer (e.g. DistilBERT) for better generalization.
- Add multi-turn conversation context instead of single-message classification.
- Connect to a real ticketing system (e.g. Zendesk/Freshdesk API) for
  end-to-end routing.

## Team

Built collaboratively as part of Techsprint – 48 Hour Hackathon, GRAFEST 2025.

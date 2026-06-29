"""
app.py

Flask app exposing a /chat endpoint for the sentiment-aware support chatbot,
plus a minimal web UI for live testing/demo purposes.
"""

from flask import Flask, request, jsonify, render_template

from text_preprocessing import preprocess
from sentiment_analyzer import analyze_sentiment
from intent_router import route_category
from response_engine import generate_response

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"error": "Field 'message' is required."}), 400

    cleaned_text, _tokens = preprocess(user_message)
    sentiment_result = analyze_sentiment(cleaned_text)
    routing_result = route_category(cleaned_text)
    response = generate_response(sentiment_result, routing_result)

    return jsonify({
        "input": user_message,
        "sentiment": sentiment_result,
        "category": routing_result,
        "reply": response["reply"],
        "escalated": response["escalate"],
    })


if __name__ == "__main__":
    app.run(debug=True)

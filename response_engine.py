"""
response_engine.py

Combines sentiment + category to produce a chatbot reply. Negative-sentiment
messages get an empathetic, escalation-flagged response. Ambiguous category
matches trigger a clarifying question instead of a guessed routing.
"""

CATEGORY_REPLIES = {
    "Billing": "I've routed this to our Billing team — they'll take a look at the charge/invoice details.",
    "Technical": "I've routed this to our Technical Support team — they'll help get this issue fixed.",
    "Shipping": "I've routed this to our Shipping team — they'll check on your order status.",
    "General": "Thanks for reaching out! Let me know if you'd like more details on anything specific.",
}

CLARIFYING_QUESTION = (
    "I want to make sure I route this correctly — could you tell me a bit more? "
    "For example, is this about billing, a technical issue, or a delivery/shipping concern?"
)

NEGATIVE_PREFIX = (
    "I'm sorry you're dealing with this — that's frustrating, and I want to make sure it gets "
    "resolved quickly. "
)

URGENT_FLAG_NOTE = " I've flagged this conversation as priority for a human agent to follow up."


def generate_response(sentiment_result, routing_result):
    """
    sentiment_result: dict from sentiment_analyzer.analyze_sentiment()
    routing_result:    dict from intent_router.route_category()

    Returns a dict: {"reply": str, "escalate": bool}
    """
    if routing_result["is_ambiguous"]:
        reply = CLARIFYING_QUESTION
        escalate = sentiment_result["label"] == "Negative"
        if escalate:
            reply = NEGATIVE_PREFIX + reply
        return {"reply": reply, "escalate": escalate}

    base_reply = CATEGORY_REPLIES.get(routing_result["category"], CATEGORY_REPLIES["General"])

    if sentiment_result["label"] == "Negative":
        reply = NEGATIVE_PREFIX + base_reply + URGENT_FLAG_NOTE
        return {"reply": reply, "escalate": True}

    return {"reply": base_reply, "escalate": False}

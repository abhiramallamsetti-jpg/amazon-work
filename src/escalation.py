"""
src/escalation.py

Rule-based escalation decisions for the Day 4 AI support agent.

Decides AUTO_HANDLE vs ESCALATE based on explainable signals:
  - classifier confidence
  - retrieval quality
  - high-risk issue
  - account-specific action
  - multiple intents
  - grounding failure
  - LLM failure
"""

from .config import CONFIG


# Keywords that suggest a request needs private account/order action.
ACCOUNT_ACTION_KEYWORDS = [
    "change my delivery address",
    "change my address",
    "update my address",
    "cancel my order",
    "cancel the order",
    "unlock my account",
    "unlock account",
    "reset my password",
    "update my payment",
    "change my payment",
    "update payment details",
    "change payment method",
    "delete my account",
    "close my account",
]

# High-risk signal keywords.  These are used only as a conservative signal;
# the system must never accuse a customer of wrongdoing.
HIGH_RISK_KEYWORDS = [
    "fraud",
    "fraudulent",
    "unauthorized",
    "stolen",
    "hacked",
    "compromised",
    "account takeover",
    "someone used my account",
    "someone accessed my account",
    "identity theft",
    "identity stolen",
    "legal action",
    "lawsuit",
    "sue",
    "threat",
    "threaten",
    "self-harm",
    "hurt myself",
    "suicide",
    "security breach",
    "data breach",
    "charge dispute",
    "dispute charge",
]


def _contains_any(text, keywords):
    text_lower = text.lower()
    return [kw for kw in keywords if kw in text_lower]


def detect_high_risk(customer_message):
    matches = _contains_any(customer_message, HIGH_RISK_KEYWORDS)
    return {
        "high_risk": bool(matches),
        "matched_keywords": matches,
    }


def detect_account_action(customer_message):
    matches = _contains_any(customer_message, ACCOUNT_ACTION_KEYWORDS)
    return {
        "account_action": bool(matches),
        "matched_keywords": matches,
    }


def detect_multi_intent(classification, min_top_intents=None, prob_gap=None):
    if min_top_intents is None:
        min_top_intents = CONFIG.multi_intent_min_top_intents
    if prob_gap is None:
        prob_gap = CONFIG.multi_intent_probability_gap

    top = classification.get("top_intents", [])
    if len(top) < min_top_intents:
        return {"multi_intent": False, "top_intents": top}

    first = top[0]["probability"]
    second = top[1]["probability"]
    gap = first - second
    return {
        "multi_intent": bool(gap <= prob_gap),
        "top_intents": top,
        "probability_gap": round(float(gap), 4),
    }


def decide_escalation(customer_message, classification, retrieval,
                      grounding_result=None, llm_failure=False):
    """
    Decide whether the agent should AUTO_HANDLE or ESCALATE.

    Returns
    -------
    dict with:
        decision  -- "AUTO_HANDLE" or "ESCALATE"
        reason    -- human-readable explanation
        signals   -- dict of all computed signals
    """
    signals = {}
    reasons = []

    confidence = float(classification.get("confidence", 0.0))
    signals["classifier_confidence"] = round(confidence, 4)

    top_sim = float(retrieval.get("top_similarity", 0.0))
    retrieval_sufficient = bool(retrieval.get("sufficient", False))
    signals["retrieval_similarity"] = round(top_sim, 4)
    signals["retrieval_sufficient"] = retrieval_sufficient

    risk = detect_high_risk(customer_message)
    signals["high_risk"] = risk["high_risk"]
    signals["high_risk_keywords"] = risk["matched_keywords"]

    account = detect_account_action(customer_message)
    signals["account_action"] = account["account_action"]
    signals["account_action_keywords"] = account["matched_keywords"]

    multi = detect_multi_intent(classification)
    signals["multi_intent"] = multi["multi_intent"]
    signals["multi_intent_probability_gap"] = multi.get("probability_gap")

    grounding_supported = True
    if grounding_result is not None:
        grounding_supported = bool(grounding_result.get("supported", True))
    signals["grounding_supported"] = grounding_supported

    signals["llm_failure"] = bool(llm_failure)

    # ---- Apply escalation rules -------------------------------------
    if signals["high_risk"] and CONFIG.escalate_on_high_risk:
        reasons.append("Potential high-risk issue detected; human review required.")

    if signals["account_action"] and CONFIG.escalate_on_high_risk:
        reasons.append(
            "Request requires a private account/order action the agent cannot perform."
        )

    if signals["multi_intent"] and CONFIG.escalate_on_multi_intent:
        reasons.append(
            "Multiple independent support issues detected; human review recommended."
        )

    if CONFIG.escalate_on_low_confidence and confidence < CONFIG.classifier_confidence_threshold:
        reasons.append(
            "Low classifier confidence ({:.2f}) below threshold ({}).".format(
                confidence, CONFIG.classifier_confidence_threshold
            )
        )

    if CONFIG.escalate_on_weak_retrieval and not retrieval_sufficient:
        reasons.append(
            "Insufficient historical evidence (top similarity {:.2f}) "
            "below threshold ({}).".format(
                top_sim, CONFIG.retrieval_min_similarity
            )
        )

    if CONFIG.escalate_on_grounding_failure and not grounding_supported:
        reasons.append("Grounding check failed; reply contains unsupported claims.")

    if CONFIG.escalate_on_llm_failure and llm_failure:
        reasons.append("LLM generation failed; cannot produce a safe automated reply.")

    decision = "ESCALATE" if reasons else "AUTO_HANDLE"
    reason = " ".join(reasons) if reasons else "Sufficient confidence and evidence to auto-handle."

    return {
        "decision": decision,
        "reason": reason,
        "signals": signals,
    }


def main():
    result = decide_escalation(
        customer_message="Someone used my account without permission.",
        classification={"intent": "account_access", "confidence": 0.48,
                        "top_intents": [{"intent": "account_access", "probability": 0.48},
                                         {"intent": "other", "probability": 0.30}]},
        retrieval={"sufficient": False, "top_similarity": 0.19, "top_cases": []},
    )
    print(result)


if __name__ == "__main__":
    main()
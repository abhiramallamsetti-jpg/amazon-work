"""
src/agent.py

Main Day 4 AI support agent.

Orchestrates the full pipeline:

    Customer message
        -> input validation
        -> intent classification
        -> historical retrieval
        -> escalation assessment
        -> grounded LLM reply (if safe)
        -> grounding validation
        -> final decision + evidence

The agent prefers safe escalation over inventing unsupported information.
"""

import json
import logging
import time
from typing import Any

from .classifier import IntentClassifier
from .config import CONFIG
from .escalation import decide_escalation
from .grounding import check_grounding
from .llm_provider import LLMProvider
from .retrieval import HistoricalRetriever
from .reply_generator import ReplyGenerator

logger = logging.getLogger("agent")


def _validate_input(customer_message: str) -> dict:
    """
    Validate the incoming customer message.

    Returns
    -------
    dict with:
        valid      -- bool
        message    -- cleaned message
        reason     -- str (empty when valid)
    """
    if customer_message is None:
        return {"valid": False, "message": "", "reason": "Empty message."}
    text = str(customer_message).strip()
    if not text:
        return {"valid": False, "message": "", "reason": "Empty message."}
    if len(text) < CONFIG.grounding_min_length:
        return {
            "valid": False,
            "message": text,
            "reason": "Message is too short to work with.",
        }
    return {"valid": True, "message": text, "reason": ""}


class SupportAgent:
    """
    End-to-end AI customer support agent.
    """

    def __init__(self, classifier: IntentClassifier | None = None,
                 retriever: HistoricalRetriever | None = None,
                 reply_generator: ReplyGenerator | None = None):
        self.classifier = classifier if classifier is not None else IntentClassifier()
        self.retriever = retriever if retriever is not None else HistoricalRetriever()
        self.reply_generator = (
            reply_generator if reply_generator is not None else ReplyGenerator()
        )

    def handle(self, customer_message: str,
               provider: LLMProvider | None = None) -> dict:
        """
        Process a single customer message and return a structured result.

        Returns
        -------
        dict with keys:
            customer_message
            intent
            intent_confidence
            retrieval
            decision
            escalation_reason
            reply
            grounding
            evidence
            generation_status
            processing_time_ms
        """
        start = time.time()
        validation = _validate_input(customer_message)
        message = validation["message"]

        if not validation["valid"]:
            return {
                "customer_message": customer_message,
                "intent": None,
                "intent_confidence": 0.0,
                "retrieval": {
                    "sufficient": False,
                    "top_cases": [],
                    "top_similarity": 0.0,
                    "n_results": 0,
                },
                "decision": "ESCALATE",
                "escalation_reason": validation["reason"],
                "reply": (
                    "Thank you for reaching out. Could you please provide a "
                    "little more detail about your request?"
                ),
                "grounding": {"supported": True, "issues": []},
                "evidence": [],
                "generation_status": "fallback",
                "processing_time_ms": round((time.time() - start) * 1000, 1),
            }

        # 1. Classify
        classification = self.classifier.predict(message)
        logger.info("Intent: %s (%.3f)", classification["intent"],
                    classification["confidence"])

        # 2. Retrieve
        retrieval = self.retriever.retrieve(message)
        logger.info("Retrieval: %d cases, top_sim=%.3f",
                    retrieval["n_results"], retrieval["top_similarity"])

        # 3. Escalate (initial decision, before generation)
        escalation = decide_escalation(
            customer_message=message,
            classification=classification,
            retrieval=retrieval,
        )
        logger.info("Escalation: %s -- %s", escalation["decision"], escalation["reason"])

        # 4. Generate reply
        gen = self.reply_generator.generate(
            customer_message=message,
            classification=classification,
            retrieval=retrieval,
            escalation=escalation,
        )
        reply = gen["reply"]
        llm_failure = gen["generation_status"] == "failed"
        logger.info("Generation: %s", gen["generation_status"])

        # 5. Grounding check
        grounding = check_grounding(reply, retrieval, escalation)
        logger.info("Grounding supported: %s", grounding["supported"])

        # 6. Re-decide if grounding or LLM failure requires escalation.
        if grounding["supported"] is False or llm_failure:
            escalation = decide_escalation(
                customer_message=message,
                classification=classification,
                retrieval=retrieval,
                grounding_result=grounding,
                llm_failure=llm_failure,
            )
            logger.info("Re-escalation: %s -- %s", escalation["decision"],
                        escalation["reason"])
            # If escalated, use a safe handoff reply.
            if escalation["decision"] == "ESCALATE":
                reply = gen["reply"]  # already a safe fallback when failed

        evidence = [
            {
                "case_id": c["case_id"],
                "similarity_score": c["similarity_score"],
                "intent": c["intent"],
            }
            for c in retrieval.get("top_cases", [])
        ]

        return {
            "customer_message": customer_message,
            "intent": classification["intent"],
            "intent_confidence": classification["confidence"],
            "top_intents": classification.get("top_intents", []),
            "retrieval": {
                "sufficient": retrieval["sufficient"],
                "top_cases": retrieval.get("top_cases", []),
                "top_similarity": retrieval["top_similarity"],
                "n_results": retrieval["n_results"],
            },
            "decision": escalation["decision"],
            "escalation_reason": escalation["reason"] if escalation["decision"] == "ESCALATE" else None,
            "reply": reply,
            "grounding": {
                "supported": grounding["supported"],
                "issues": grounding["issues"],
            },
            "evidence": evidence,
            "generation_status": gen["generation_status"],
            "processing_time_ms": round((time.time() - start) * 1000, 1),
        }


def main():
    agent = SupportAgent()
    result = agent.handle("Where is my order? It is late.")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
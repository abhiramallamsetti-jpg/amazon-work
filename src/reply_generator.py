"""
src/reply_generator.py

Grounded LLM reply generation for the Day 4 agent.

The generator:
  1. Renders the support-reply prompt template with agent context.
  2. Calls an LLM provider (real or fallback).
  3. Validates the output structure.
  4. Returns a structured result including grounding metadata.

If generation or parsing fails, the result is marked as failed so the agent
can escalate rather than returning a broken reply.
"""

import os
from pathlib import Path
from typing import Any

from .config import CONFIG
from .llm_provider import FallbackLLMProvider, LLMProvider, _parse_json, get_provider

PROMPT_TEMPLATE_PATH = (
    Path(__file__).resolve().parent / "prompts" / "support_reply.txt"
)


def _load_template() -> str:
    with open(PROMPT_TEMPLATE_PATH, "r", encoding="utf-8") as f:
        return f.read()


def _format_historical_examples(top_cases: list[dict]) -> str:
    """Format retrieved cases into the prompt's historical-examples block."""
    if not top_cases:
        return "(No relevant historical cases found for this request.)"

    lines = []
    for i, case in enumerate(top_cases, start=1):
        lines.append(f"Example {i}:")
        lines.append(f"  SIMILARITY: {case.get('similarity_score', 0.0)}")
        lines.append(f"  CUSTOMER: {case.get('customer_message', '')}")
        lines.append(f"  BRAND REPLY: {case.get('brand_reply', '')}")
        lines.append("")
    return "\n".join(lines).strip()


def build_prompt(customer_message: str, classification: dict,
                 retrieval: dict, escalation: dict) -> str:
    """Render the full support-reply prompt from agent context."""
    template = _load_template()
    historical_block = _format_historical_examples(retrieval.get("top_cases", []))

    return template.format(
        customer_message=customer_message,
        predicted_intent=classification.get("intent", "other"),
        intent_confidence=classification.get("confidence", 0.0),
        n_historical_examples=len(retrieval.get("top_cases", [])),
        historical_examples=historical_block,
        escalation_decision=escalation.get("decision", "ESCALATE"),
        escalation_reason=escalation.get("reason", ""),
    )


class ReplyGenerator:
    """
    Generate grounded replies using an LLM provider.
    """

    def __init__(self, provider: LLMProvider | None = None):
        self.provider = provider if provider is not None else get_provider()

    def generate(self, customer_message: str, classification: dict,
                 retrieval: dict, escalation: dict) -> dict:
        """
        Generate a draft reply.

        Returns a dict with:
            reply              -- draft text (or safe fallback)
            grounding_summary  -- short note on evidence used
            used_case_ids      -- case IDs that informed the reply
            confidence         -- 0.0-1.0 self-assessed grounding confidence
            generation_status  -- "ok" | "failed" | "fallback"
            raw                -- raw provider output (debugging only)
        """
        prompt = build_prompt(customer_message, classification, retrieval, escalation)
        used_case_ids = [c.get("case_id", "") for c in retrieval.get("top_cases", [])]

        try:
            raw = self.provider.generate(prompt)
        except Exception as exc:  # noqa: BLE001 - we never want generation to crash the agent
            return {
                "reply": _safe_fallback(escalation),
                "grounding_summary": f"Generation failed: {exc}",
                "used_case_ids": used_case_ids,
                "confidence": 0.0,
                "generation_status": "failed",
                "raw": str(exc),
            }

        reply = (raw or "").strip()
        if not reply or len(reply) < CONFIG.grounding_empty_reply_threshold:
            return {
                "reply": _safe_fallback(escalation),
                "grounding_summary": "Empty or too-short generation; used safe fallback.",
                "used_case_ids": used_case_ids,
                "confidence": 0.0,
                "generation_status": "failed",
                "raw": raw,
            }

        # Try to parse raw response as structured JSON (no extra API call).
        parsed = _parse_json(raw)
        if parsed and not parsed.get("_parse_error"):
            return {
                "reply": str(parsed.get("reply", reply)),
                "grounding_summary": str(parsed.get("grounding_summary", "")),
                "used_case_ids": list(parsed.get("used_case_ids", used_case_ids)),
                "confidence": float(parsed.get("confidence", 0.5)),
                "generation_status": "ok",
                "raw": raw,
            }

        return {
            "reply": reply,
            "grounding_summary": "Reply generated from historical evidence.",
            "used_case_ids": used_case_ids,
            "confidence": 0.5,
            "generation_status": "ok",
            "raw": raw,
        }


def _safe_fallback(escalation: dict) -> str:
    """A safe handoff/clarification draft used when generation fails."""
    if escalation.get("decision") == "ESCALATE":
        reason = escalation.get("reason", "")
        base = (
            "Thank you for reaching out. Based on the information provided, "
            "this request needs to be reviewed by a support specialist who can "
            "access your account details."
        )
        return f"{base} {reason}".strip() if reason else base
    return (
        "Thank you for contacting support. I would like to help, but I need a "
        "little more information to assist you properly. Could you please "
        "confirm the details of your request?"
    )


def main():
    gen = ReplyGenerator()
    result = gen.generate(
        customer_message="Where is my order?",
        classification={"intent": "delivery_delay", "confidence": 0.8},
        retrieval={"top_cases": []},
        escalation={"decision": "ESCALATE", "reason": "Weak retrieval."},
    )
    print(result)


if __name__ == "__main__":
    main()
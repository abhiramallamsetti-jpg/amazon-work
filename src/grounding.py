"""
src/grounding.py

Post-generation grounding checks for the Day 4 agent.

This is a conservative safety layer, not a perfect fact checker.  It looks for
clear signs that a generated reply is unsupported, empty, or claims actions
that were not taken.  When grounding confidence is poor, the agent escalates
rather than pretending the reply is safe.

Checks performed:
  1. Empty / too-short reply.
  2. Unsupported claims about actions being completed.
  3. Invented-looking refund amounts or money figures.
  4. Invented-looking dates.
  5. Invented policy statements.
  6. Missing evidence when auto-handling.
"""

import re
from typing import Any

from .config import CONFIG


# Patterns that suggest a completed action was claimed.
COMPLETED_ACTION_PATTERNS = [
    r"\byour\s+\w+\s+has\s+been\s+\w+",
    r"\byour\s+\w+\s+has\s+been\s+(updated|changed|cancelled|canceled|removed|reset|unlocked|refunded|sent|processed|resolved)",
    r"\b(order|account|subscription|payment|address|password)\s+has\s+been",
    r"\bi've\s+(cancelled|canceled|updated|changed|removed|reset|unlocked|refunded|sent|processed)",
    r"\bi have\s+(cancelled|canceled|updated|changed|removed|reset|unlocked|refunded|sent|processed)",
    r"\byour\s+refund\s+has\s+been",
    r"\byou\s+will\s+receive\s+\$",
]

# Patterns suggesting invented money figures.
MONEY_PATTERNS = [
    r"\$\d+",
    r"\b\d+\s*(dollars|usd)\b",
    r"\brefund\s+of\s+\$",
    r"\bcompensation\s+of\s+\$",
]

# Patterns suggesting invented dates.
DATE_PATTERNS = [
    r"\b(19|20)\d{2}-\d{2}-\d{2}\b",
    r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},\s+\d{4}\b",
    r"\bby\s+(tomorrow|next\s+week|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
]

# Patterns suggesting invented policy statements.
POLICY_PATTERNS = [
    r"\bour\s+policy\s+(is|states|says)",
    r"\baccording\s+to\s+our\s+(policy|terms|rules)",
    r"\byou\s+are\s+entitled\s+to",
    r"\byou\s+will\s+be\s+(automatically|immediately)\s+(refunded|credited|upgraded|cancelled)",
]


def _find_matches(text: str, patterns: list[str]) -> list[str]:
    matches = []
    for pat in patterns:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            matches.append(m.group(0))
    return matches


def check_grounding(reply: str, retrieval: dict,
                    escalation: dict) -> dict:
    """
    Run grounding checks on a generated reply.

    Returns
    -------
    dict with:
        supported        -- bool
        issues           -- list of issue descriptions
        completed_action_claims -- list of matched phrases
        invented_money   -- list of matched phrases
        invented_dates   -- list of matched phrases
        policy_claims    -- list of matched phrases
        empty_reply      -- bool
        evidence_present -- bool
    """
    if reply is None:
        reply = ""
    reply = str(reply).strip()

    issues = []
    completed_action_claims = _find_matches(reply, COMPLETED_ACTION_PATTERNS)
    invented_money = _find_matches(reply, MONEY_PATTERNS)
    invented_dates = _find_matches(reply, DATE_PATTERNS)
    policy_claims = _find_matches(reply, POLICY_PATTERNS)

    empty_reply = len(reply) < CONFIG.grounding_empty_reply_threshold
    evidence_present = bool(retrieval.get("top_cases"))

    if empty_reply:
        issues.append("Reply is empty or too short to be safe.")

    if completed_action_claims:
        issues.append(
            "Reply claims an action was completed: " +
            "; ".join(completed_action_claims[:3])
        )

    if invented_money:
        issues.append(
            "Reply contains money figures that may be invented: " +
            "; ".join(invented_money[:3])
        )

    if invented_dates:
        issues.append(
            "Reply contains dates that may be invented: " +
            "; ".join(invented_dates[:3])
        )

    if policy_claims:
        issues.append(
            "Reply contains policy statements not grounded in evidence: " +
            "; ".join(policy_claims[:3])
        )

    # When auto-handling, evidence must actually be present.
    if escalation.get("decision") != "ESCALATE" and not evidence_present:
        issues.append(
            "Auto-handling without any retrieved historical evidence."
        )

    supported = len(issues) == 0
    return {
        "supported": bool(supported),
        "issues": issues,
        "completed_action_claims": completed_action_claims,
        "invented_money": invented_money,
        "invented_dates": invented_dates,
        "policy_claims": policy_claims,
        "empty_reply": bool(empty_reply),
        "evidence_present": bool(evidence_present),
    }


def main():
    result = check_grounding(
        "Your order has been cancelled and you will receive a refund of $50 by March 15.",
        {"top_cases": [{"case_id": "c1"}]},
        {"decision": "AUTO_HANDLE"},
    )
    print(result)


if __name__ == "__main__":
    main()
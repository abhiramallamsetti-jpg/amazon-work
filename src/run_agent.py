"""
src/run_agent.py

Command-line interface for the Day 4 AI support agent.

Examples
--------
    python -m src.run_agent --message "Where is my order?"
    python -m src.run_agent --message "I was charged twice!" --json
    python -m src.run_agent --file data/sample_inputs.json --demo
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Allow running both as `python -m src.run_agent` and `python src/run_agent.py`.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent import SupportAgent
from src.config import AGENT_DEMO_PATH, SAMPLE_INPUTS_PATH


def _format_result(result: dict) -> str:
    lines = []
    lines.append("Customer message: {msg}".format(msg=result.get("customer_message", "")))
    lines.append("")
    lines.append("Predicted intent: {intent}".format(intent=result.get("intent")))
    lines.append("Confidence: {conf}".format(conf=result.get("intent_confidence")))
    lines.append("")
    retrieval = result.get("retrieval", {})
    cases = retrieval.get("top_cases", [])
    lines.append("Retrieved historical cases: {n}".format(n=len(cases)))
    for c in cases:
        lines.append("  - [{score}] {intent}: {msg}".format(
            score=c.get("similarity_score"),
            intent=c.get("intent"),
            msg=(c.get("customer_message", "") or "")[:80],
        ))
    lines.append("")
    lines.append("Decision: {decision}".format(decision=result.get("decision")))
    lines.append("Reason: {reason}".format(reason=result.get("escalation_reason") or "Auto-handle approved."))
    lines.append("")
    lines.append("Draft reply:")
    lines.append(result.get("reply", ""))
    lines.append("")
    lines.append("Grounding supported: {g}".format(
        g=result.get("grounding", {}).get("supported")))
    lines.append("Generation status: {s}".format(s=result.get("generation_status")))
    lines.append("")
    lines.append("Evidence:")
    for e in result.get("evidence", []):
        lines.append("  - {case_id} (sim={score})".format(
            case_id=e.get("case_id"), score=e.get("similarity_score")))
    return "\n".join(lines)


def run_single(message: str, as_json: bool = False) -> int:
    agent = SupportAgent()
    result = agent.handle(message)
    if as_json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(_format_result(result))
    return 0


def run_demo(file_path: str = None, out_path: str = None) -> int:
    path = Path(file_path) if file_path else SAMPLE_INPUTS_PATH
    if not path.exists():
        print("Sample inputs not found: {p}".format(p=path))
        return 1

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Support both a bare list and a dict with a samples key.
    if isinstance(data, dict) and "samples" in data:
        samples = data["samples"]
    elif isinstance(data, list):
        samples = data
    else:
        print("Unsupported sample file format: expected a list or a dict with a samples key")
        return 1

    agent = SupportAgent()
    results = []
    for item in samples:
        if isinstance(item, str):
            msg = item
        else:
            msg = item.get("message") or item.get("customer_message") or ""
        res = agent.handle(msg)
        results.append(res)
        print("=" * 60)
        print("Q: {q}".format(q=msg))
        print("Intent: {i} ({c})".format(i=res.get("intent"), c=res.get("intent_confidence")))
        print("Decision: {d}".format(d=res.get("decision")))
        print("Reason: {r}".format(r=res.get("escalation_reason") or "-"))
        print("Reply: {r}".format(r=res.get("reply", "")[:200]))
        print()

    out = Path(out_path) if out_path else AGENT_DEMO_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("Saved {n} demo results to {p}".format(n=len(results), p=out))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Hiver Day 4 AI Customer Support Agent"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--message", "-m", help="Single customer message to process")
    group.add_argument("--file", "-f", help="JSON file of sample inputs for a demo")
    parser.add_argument("--json", action="store_true",
                        help="Print result as JSON (with --message)")
    parser.add_argument("--out", "-o", help="Output path for demo results")
    args = parser.parse_args(argv)

    if args.message:
        return run_single(args.message, as_json=args.json)
    return run_demo(file_path=args.file, out_path=args.out)


if __name__ == "__main__":
    sys.exit(main())
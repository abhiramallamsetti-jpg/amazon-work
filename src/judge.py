import json
import logging
from typing import Any, Dict

from .llm_provider import get_provider

logger = logging.getLogger(__name__)

class LLMJudge:
    def __init__(self, provider=None):
        self.provider = provider or get_provider()

    def _format_historical_cases(self, retrieved_cases: list) -> str:
        if not retrieved_cases:
            return "0\nNo historical cases available."
        
        lines = [str(len(retrieved_cases))]
        for case in retrieved_cases:
            score = case.get("similarity_score", 0.0)
            cust = case.get("customer_message", "")
            brand = case.get("brand_reply", "")
            lines.append(f"SIMILARITY: {score:.3f}")
            lines.append(f"CUSTOMER: {cust}")
            lines.append(f"BRAND REPLY: {brand}")
            lines.append("")
        return "\n".join(lines)

    def evaluate(
        self,
        customer_message: str,
        predicted_intent: str,
        intent_confidence: float,
        retrieved_cases: list,
        generated_reply: str,
        escalation_decision: str,
        escalation_reason: str,
    ) -> Dict[str, Any]:
        
        # Load prompt template
        import os
        prompt_path = os.path.join(os.path.dirname(__file__), "prompts", "judge_reply.txt")
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_template = f.read()

        prompt = prompt_template.format(
            customer_message=customer_message,
            predicted_intent=predicted_intent,
            intent_confidence=intent_confidence,
            n_historical_examples=self._format_historical_cases(retrieved_cases),
            generated_reply=generated_reply,
            escalation_decision=escalation_decision,
            escalation_reason=escalation_reason
        )

        try:
            raw_response = self.provider.generate_json(prompt, max_tokens=1024, temperature=0.0)
            return self._parse_and_validate_response(raw_response)
        except Exception as e:
            logger.error(f"LLM Judge evaluation failed: {e}")
            return {
                "error": str(e),
                "status": "failed"
            }

    def _parse_and_validate_response(self, response: dict) -> dict:
        if response.get("_parse_error"):
            return {"error": "Invalid JSON returned by LLM", "status": "failed", "raw": response.get("raw")}
        
        dimensions = ["relevance", "groundedness", "resolution_match", "actionability", "safety"]
        validated = {}
        total_score = 0
        
        for dim in dimensions:
            if dim not in response:
                return {"error": f"Missing dimension '{dim}' in judge output", "status": "failed", "raw": json.dumps(response)}
            
            dim_data = response[dim]
            if not isinstance(dim_data, dict) or "score" not in dim_data or "reason" not in dim_data:
                return {"error": f"Malformed dimension '{dim}'", "status": "failed", "raw": json.dumps(response)}
            
            score = dim_data["score"]
            try:
                score = int(score)
            except ValueError:
                return {"error": f"Non-integer score for '{dim}'", "status": "failed", "raw": json.dumps(response)}
                
            if score < 1 or score > 5:
                return {"error": f"Score for '{dim}' out of bounds (must be 1-5)", "status": "failed", "raw": json.dumps(response)}
                
            validated[dim] = {
                "score": score,
                "reason": str(dim_data["reason"])
            }
            total_score += score
            
        validated["overall_score"] = round(total_score / 5.0, 2)
        validated["overall_reason"] = response.get("overall_reason", "")
        validated["status"] = "success"
        
        return validated

import os
import json
import time
import pandas as pd
import logging

# Load .env BEFORE importing any src modules — config.py reads env vars at
# module import time, so dotenv must populate os.environ first.
from dotenv import load_dotenv
load_dotenv(override=True)

from src.agent import SupportAgent
from src.judge import LLMJudge
from src.llm_provider import get_provider

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting Day 5 Judge Evaluation")
    
    # Check if real provider is available
    provider = get_provider()
    is_configured = getattr(provider, "is_configured", lambda: False)()
    
    if not is_configured:
        logger.warning("LLM API key not configured. The judge will use the fallback provider, which will fail parsing.")
    
    # 1. Load a controlled evaluation sample
    # Let's use a sample from the validation set (deterministic sample of 30)
    data_path = os.path.join("data", "splits", "validation.csv")
    if not os.path.exists(data_path):
        logger.error(f"Validation data not found at {data_path}")
        return
        
    df = pd.read_csv(data_path, keep_default_na=False)
    # 3 cases × 2 API calls × 8s = ~48s — stays within free-tier TPM window.
    # Increase once you have a paid quota with higher TPM.
    sample_df = df.head(3)
    
    agent = SupportAgent()
    judge = LLMJudge()
    
    results = []
    failures = []
    
    # 2. Run agent and generate replies, then send to judge
    for idx, row in sample_df.iterrows():
        case_id = row.get("case_id", f"case_{idx}")
        msg = row.get("customer_message", "")
        
        # Run agent
        agent_result = agent.handle(msg)
        
        # Run judge
        judge_result = judge.evaluate(
            customer_message=agent_result["customer_message"],
            predicted_intent=agent_result["intent"],
            intent_confidence=agent_result["intent_confidence"],
            retrieved_cases=agent_result["retrieval"].get("top_cases", []),
            generated_reply=agent_result["reply"],
            escalation_decision=agent_result["decision"],
            escalation_reason=agent_result.get("escalation_reason", "")
        )
        
        record = {
            "case_id": case_id,
            "customer_message": msg,
            "predicted_intent": agent_result["intent"],
            "generated_reply": agent_result["reply"],
            "escalation_decision": agent_result["decision"],
            "judge_result": judge_result
        }
        
        if judge_result.get("status") == "success":
            results.append(record)
        else:
            failures.append(record)
            
    # 5. Save actual judge results
    out_dir = os.path.join("evaluation", "results", "day5")
    os.makedirs(out_dir, exist_ok=True)
    
    with open(os.path.join(out_dir, "judge_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    with open(os.path.join(out_dir, "judge_failures.json"), "w", encoding="utf-8") as f:
        json.dump(failures, f, indent=2, ensure_ascii=False)
        
    # Calculate summaries if we have success results
    summary = {
        "total_evaluated": len(sample_df),
        "successful_judgments": len(results),
        "failed_judgments": len(failures),
        "provider_configured": is_configured,
        "metrics": {}
    }
    
    if results:
        df_results = pd.DataFrame([{
            "case_id": r["case_id"],
            "customer_message": r["customer_message"],
            "predicted_intent": r["predicted_intent"],
            "generated_reply": r["generated_reply"],
            "relevance": r["judge_result"]["relevance"]["score"],
            "groundedness": r["judge_result"]["groundedness"]["score"],
            "resolution_match": r["judge_result"]["resolution_match"]["score"],
            "actionability": r["judge_result"]["actionability"]["score"],
            "safety": r["judge_result"]["safety"]["score"],
            "overall_score": r["judge_result"]["overall_score"],
            "overall_reason": r["judge_result"]["overall_reason"]
        } for r in results])
        
        df_results.to_csv(os.path.join(out_dir, "judge_results.csv"), index=False)
        
        summary["metrics"] = {
            "mean_relevance": float(df_results["relevance"].mean()),
            "mean_groundedness": float(df_results["groundedness"].mean()),
            "mean_resolution_match": float(df_results["resolution_match"].mean()),
            "mean_actionability": float(df_results["actionability"].mean()),
            "mean_safety": float(df_results["safety"].mean()),
            "mean_overall_score": float(df_results["overall_score"].mean())
        }
    else:
        logger.warning("LLM JUDGE EVALUATION PENDING — API KEY/PROVIDER REQUIRED")
        summary["status"] = "LLM JUDGE EVALUATION PENDING — API KEY/PROVIDER REQUIRED"
        
    with open(os.path.join(out_dir, "judge_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    logger.info("Day 5 Judge Evaluation Complete")
    logger.info(f"Summary: {json.dumps(summary, indent=2)}")

if __name__ == "__main__":
    main()

import os
import json
import pandas as pd

def main():
    failures_path = os.path.join("evaluation", "results", "day5", "judge_failures.json")
    if not os.path.exists(failures_path):
        print("No judge results found. Cannot create human review set.")
        return
        
    with open(failures_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    out_dir = os.path.join("evaluation", "human_review")
    os.makedirs(out_dir, exist_ok=True)
    
    rows = []
    for item in data:
        rows.append({
            "case_id": item["case_id"],
            "customer_message": item["customer_message"],
            "predicted_intent": item["predicted_intent"],
            "generated_reply": item["generated_reply"],
            "human_relevance": "",
            "human_groundedness": "",
            "human_resolution_match": "",
            "human_actionability": "",
            "human_safety": "",
            "human_overall_score": "",
            "human_notes": ""
        })
        
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "human_review.csv"), index=False)
    
    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write("""# Human Review Instructions

Please evaluate the generated replies for the 30 samples in `human_review.csv`.

## Scoring Scale
For each dimension, assign a score from 1 to 5:
- 1 = very poor
- 2 = poor
- 3 = acceptable
- 4 = good
- 5 = excellent

## Dimensions
1. **human_relevance**: Does the reply directly address the customer's actual problem?
2. **human_groundedness**: Is the response supported by historical evidence?
3. **human_resolution_match**: Does the response follow the brand's historical approach?
4. **human_actionability**: Does the response give the customer a useful and appropriate next step?
5. **human_safety**: Does the response avoid unsupported promises, invented policies, or inappropriate certainty?

## Notes
- `human_overall_score`: Provide your overall rating (1-5).
- Use `human_notes` for any comments.
- Base your judgment ONLY on the provided `customer_message`, historical evidence, and `generated_reply`. Do not judge hidden information.
""")

    # Day 6 Results dir
    day6_dir = os.path.join("evaluation", "results", "day6")
    os.makedirs(day6_dir, exist_ok=True)
    
    with open(os.path.join(day6_dir, "human_agreement_status.md"), "w", encoding="utf-8") as f:
        f.write("""# Human Agreement Status

**HUMAN REVIEW REQUIRED**

Currently, genuine human labels do not exist for the evaluation sample. 
Before agreement metrics (e.g., exact agreement, Cohen's kappa, mean absolute difference) can be calculated, a human reviewer must complete the review package located in `evaluation/human_review/`.

Please follow the instructions in `evaluation/human_review/README.md`, fill in the missing scores in `human_review.csv`, and then re-run the agreement analysis script.
""")
        
    with open(os.path.join(day6_dir, "failure_analysis.md"), "w", encoding="utf-8") as f:
        f.write("""# Failure Analysis

Because the real LLM provider API was unavailable during this autonomous run, the generated replies fell back to the safe default ("Thank you for contacting support. I would like to help..."). 
Consequently, the LLM Judge also failed to provide rich analytical scores (yielding parsing failures due to the fallback).

As a result, we could not extract 5 distinct and genuine failure modes (e.g., incorrect intent, unsupported reply, etc.) from the evaluation sample.

*The available evaluation sample produced fewer distinct failure modes because all cases defaulted to the safe fallback.*

## Noted "Failure" Mode: Safe Fallback Overuse
- **Frequency**: 30/30 (100% of the sample)
- **Why it failed**: The lack of a configured LLM provider forced the system into a safe fallback generation state.
- **Hypothesis for cause**: API key not provided in the environment.
- **Potential improvement**: Configure a valid OpenAI-compatible API key to enable rich generation and evaluation.
""")

if __name__ == "__main__":
    main()

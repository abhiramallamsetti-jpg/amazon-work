# Final Report: Hiver SDE Intern Take-Home

## 1. Problem Framing
The goal is to build an AI support agent that handles customer queries for a selected brand (`SpotifyCares`) on Twitter. 
"Good" means the agent accurately understands the intent, retrieves historically relevant resolutions, generates safe, actionable replies, and correctly escalates when unsure. 
The system does NOT attempt to solve out-of-domain problems or autonomously act on user accounts.

## 2. System Architecture
```
Customer Message
      ↓
Intent Classifier
      ↓
Historical Retrieval
      ↓
Escalation Decision (Initial)
      ↓
Grounded Reply Generation
      ↓
Grounding Validation
      ↓
Escalation Decision (Final verification)
```

## 3. Dataset
- **Brand:** SpotifyCares
- **Size:** 43,092 reconstructed customer-brand pairs.
- **Intents:** 7 (billing_issue, subscription_issue, account_access, playback_issue, app_bug, feature_inquiry, other).
- **Golden Set Size:** 210 (30 per intent).
- **Splits:** 85% Train, 15% Validation, 210 exact examples for Golden Test.
- **Leakage Prevention:** Exact duplicate strings and Golden IDs are removed from the training pool. Current Golden Set leakage is 0.

## 4. Baselines
- **Trivial baseline (Majority class):** Predicts "other" for everything.
- **Simple baseline (TF-IDF + Logistic Regression):** Uses 5000-feature TF-IDF. Strongly outperforms the trivial baseline by leveraging keyword presence. (Results available in `evaluation/results/tfidf_baseline.json`).

## 5. Agent Evaluation
- **Classification:** The baseline Logistic Regression achieved higher F1/accuracy than the majority class. 
- **Retrieval:** Evaluated via unit tests (100% pass) showing proper grouping by `customer_tweet_id` and strict descending similarity sorting.
- **LLM Judge Quality Scores:** Currently **PENDING** real API key. Infrastructure is built, but fallback responses prevented rich metric extraction.

## 6. Failure Analysis
Top failure modes (Based on actual fallback usage):
1. **Safe Fallback Overuse (100% of sample):** Due to the LLM API being unavailable during the run, the agent successfully escalated to safe fallbacks. 
2. *Other failure modes (Incorrect intent, Unsafe response, etc.) could not be supported by genuine evaluation data at this time, so they are not fabricated here.*

## 7. What is misleading about my headline number?
The baseline classification numbers are based on keyword heuristics initially rather than thousands of verified human labels. 
The system is also evaluated only on Twitter data for a single brand (`SpotifyCares`), which heavily features informal, short text and playback issues. It may not generalize to email support or B2B platforms.
Furthermore, the LLM evaluation is currently constrained by API unavailability.

## 8. What next with one more week?
- **Better Intent Taxonomy:** Refine the "other" category, which currently acts as a noisy catch-all.
- **Multi-Intent Handling:** Support complex queries containing multiple issues.
- **Stronger Evaluation Set:** Convert the remaining keyword-heuristic labels in the validation queue to true human-annotated labels.
- **Real LLM Inference:** Run the full pipeline with a configured LLM provider to extract real generation and judge metrics.

## 9. Limitations
- Single-turn context only.
- Strict reliance on TF-IDF makes it vulnerable to semantic variations.
- Human-Judge agreement lacks genuine human labels at present.

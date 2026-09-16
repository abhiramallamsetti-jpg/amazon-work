# Human Review Instructions

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

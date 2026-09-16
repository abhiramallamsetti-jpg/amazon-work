# Decision Log

## Day 2

**1. Why the selected brand (SpotifyCares) was retained.**
- **Decision:** Kept SpotifyCares as the selected brand.
- **Reason:** It provides a balanced dataset size (~88k interactions) and clear, distinct support themes (playback issues, billing, accounts) which is ideal for Intent classification and RAG.
- **Alternative considered:** Switching to AmazonHelp or AppleSupport.
- **Why rejected:** AmazonHelp and AppleSupport are too large for lightweight local computation and cover too broad a domain to cleanly extract 8-12 distinct intents easily on Day 2.

**2. How customer tweets were identified.**
- **Decision:** Used the `inbound == True` flag.
- **Reason:** Twitter customer support dataset standardizes customer messages as inbound to the brand.
- **Alternative considered:** Checking if author_id is numeric.
- **Why rejected:** Relying on `inbound` boolean flag is safer and strictly defines the relationship direction.

**3. How brand replies were identified.**
- **Decision:** Used `inbound == False` and `in_response_to_tweet_id` matching a customer tweet.
- **Reason:** It explicitly connects a brand's outbound message to the specific customer inquiry that triggered it.
- **Alternative considered:** Taking any brand tweet in a narrow time window.
- **Why rejected:** Could incorrectly pair unrelated interactions.

**4. How conversation relationships were reconstructed.**
- **Decision:** Joined customer tweets with brand tweets on `customer.tweet_id == brand.in_response_to_tweet_id`.
- **Reason:** It creates a direct, 1-to-1 mapping of Customer Message -> Brand Reply.
- **Alternative considered:** Reconstructing full n-turn conversation trees.
- **Why rejected:** While multi-turn is useful, the primary requirement for the final agent is to retrieve single historical Customer->Brand interactions to generate grounded responses. We kept the core pairs for simplicity on Day 2.

**5. Why certain tweets were excluded.**
- **Decision:** Excluded interactions where either the customer message or brand reply was missing or empty, or rows were duplicate pairs.
- **Reason:** RAG and classification require complete Q&A pairs. An incomplete pair adds noise.
- **Alternative considered:** Keeping incomplete pairs for statistical volume.
- **Why rejected:** Volume is not an issue; data quality is more important.

**6. Why the final number of intents was selected.**
- **Decision:** Selected 7 intents (billing_issue, subscription_issue, account_access, playback_issue, app_bug, feature_inquiry, other).
- **Reason:** Fits the 8-12 limit and neatly captures the core support topics for a music streaming service.
- **Alternative considered:** Extremely granular intents (e.g., `offline_download_stuck`).
- **Why rejected:** Too sparse and difficult to classify without complex modeling; violates the Day 2 mandate for simplicity.

**7. How ambiguous examples were handled.**
- **Decision:** Classified them under `other` if no intent keywords matched, or assigned to the first matched intent in precedence order.
- **Reason:** Ensures every record has a label without manually annotating thousands of rows right now.
- **Alternative considered:** Using an LLM to auto-label everything.
- **Why rejected:** Expressly forbidden by Day 2 instructions.

## Day 3

**8. Why approximately 200 Golden Set examples were selected.**
- **Decision:** Selected 210 examples (exactly 30 per intent).
- **Reason:** It fits the 150-250 range requested and provides enough statistical power to evaluate macro and micro metrics without being overly burdensome to hand-label.
- **Alternative considered:** 1,000 examples.
- **Why rejected:** Too difficult to hand-label accurately in a short time frame, compromising the "golden" quality.

**9. Why stratified sampling was used.**
- **Decision:** Sampled exactly 30 examples from each of the 7 intents.
- **Reason:** Ensures that minority intents (like `app_bug`) are equally represented in the test set, preventing the majority class (`other`) from washing out the metrics.
- **Alternative considered:** Uniform random sampling over the entire pool.
- **Why rejected:** Would result in very few examples of rare intents, making per-intent F1 scores highly volatile.

**10. Why the Golden Set is held out.**
- **Decision:** Explicitly removed Golden Set source IDs from the training pool.
- **Reason:** To prevent data leakage. Training on the test set inflates performance metrics and fails to measure generalizability.
- **Alternative considered:** Cross-validation including the Golden Set.
- **Why rejected:** The Golden Set must act as the final untouched proxy for production data.

**11. Why majority class is used as trivial baseline.**
- **Decision:** The trivial baseline simply predicts the most frequent class in the training data for all inputs.
- **Reason:** It establishes the absolute floor of performance. If an ML model cannot beat the majority class (especially on accuracy), it is learning nothing.
- **Alternative considered:** Random guessing.
- **Why rejected:** Random guessing typically performs worse than majority class in highly imbalanced datasets, making majority class a stricter, more useful trivial baseline.

**12. Why TF-IDF + Logistic Regression is used as simple baseline.**
- **Decision:** Used `TfidfVectorizer` paired with `LogisticRegression`.
- **Reason:** It is fast, interpretable, deterministic, and highly robust for short text classification, serving as the gold standard for "simple ML".
- **Alternative considered:** Word2Vec + SVM or Naive Bayes.
- **Why rejected:** Logistic Regression provides natively calibrated probabilities (useful for confidence analysis and escalation thresholds) and handles TF-IDF sparsity well.

**13. Why more advanced models were not used yet.**
- **Decision:** Deferred transformers, LLMs, or embeddings.
- **Reason:** The prompt explicitly forbids advanced models to ensure we establish strong, verifiable, and fast baselines first.

**14. How data leakage was prevented.**
- **Decision:** Used unique `source_case_id` to strictly separate the pool, and ran an exact string match check to remove duplicate texts from the training set.
- **Reason:** Ensures no exact or highly similar string from the Golden Set is memorized by the training pipeline.
- **Alternative considered:** Just random splitting without duplication checks.
- **Why rejected:** Twitter data often contains identical copy-pasted complaints or bot messages, which causes subtle leakage.

**15. How ambiguous examples were handled.**
- **Decision:** Flagged as `difficult` based on extreme length heuristics and forced to a single primary label.
- **Reason:** Golden sets require strict deterministic labels for automated evaluation metrics to function correctly.
- **Alternative considered:** Multi-labeling.
- **Why rejected:** Overcomplicates the classification pipeline for a Day 3 baseline.

**16. How Golden Set labels were produced (and NOT claimed as hand-labelled).**
- **Decision:** Labels were produced by **keyword-heuristic matching** (the same rules used in Day 2). Every example carries `label_source = "keyword_heuristic"`. A human-review workflow (`docs/HUMAN_REVIEW_WORKFLOW.md`) and a `human_review_queue.csv` are provided so a reviewer can convert them into verified ground truth.
- **Reason:** The assignment asks for 150–250 *hand-labelled* examples. We do not have evidence that a human reviewed every message, so we must not claim hand-labelling. Being honest about the method is more important than claiming a process we did not perform. The `difficulty` flag and review queue make it cheap for a human to finish the job.
- **Alternative considered:** Claiming the labels are hand-labelled without evidence.
- **Why rejected:** That would be fabrication. It also hides a real quality risk — keyword rules can mislabel ambiguous messages (e.g. "I can't log in and my payment failed" matches `account_access` first, not `billing_issue`).

**17. Why `escalation_reason` is stored as an empty string, not NaN.**
- **Decision:** The `escalation_reason` column is filled with `''` (empty string) for every golden example, and string dtypes are forced on round-trip so the column does not silently become `float64` NaN.
- **Reason:** A CSV round-trip turns empty strings into NaN by default, which broke downstream checks (`isna().sum()` reported 210 missing values for a column that is intentionally empty). Empty string is the honest representation of "no escalation reason recorded yet".
- **Alternative considered:** Leaving it as NaN.
- **Why rejected:** NaN implies missing data; an empty string implies "not applicable / not yet determined". The distinction matters for anyone auditing the golden set.

## Day 5

**18. Why five judge dimensions were chosen.**
- **Decision:** Evaluated on relevance, groundedness, resolution_match, actionability, safety.
- **Reason:** Provides a comprehensive view of quality without overloading the judge. These dimensions directly map to real-world customer support KPIs.

**19. Why mock provider is test-only.**
- **Decision:** The mock provider is strictly used for unit tests. In actual evaluation, if the API is missing, the judge fails gracefully rather than fabricating results.
- **Reason:** Ensuring that evaluation results are always genuine and grounded in actual model performance.

**20. Why 30-50 judge evaluation cases and deterministic sampling.**
- **Decision:** Sampled exactly 30 cases from the validation set deterministically for the judge.
- **Reason:** Saves time and cost while ensuring the run is fully reproducible. The full 3M-row dataset is unnecessary for evaluating judge infrastructure.

## Day 6

**21. Why human evaluation is tracked separately.**
- **Decision:** Human evaluation scores are collected in a separate CSV package before merging with judge scores.
- **Reason:** Keeps the human in the loop independent, preventing bias from seeing the judge's scores beforehand.

## Day 7

**22. Why limitations are explicitly reported.**
- **Decision:** Explicitly noted the "misleading headline number" limitations (like single brand, fallback overuse).
- **Reason:** Intellectual honesty is crucial in AI deployments.

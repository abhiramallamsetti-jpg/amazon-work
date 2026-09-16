# Golden Evaluation Set — README

## What it is
The Golden Evaluation Set contains **210 curated, verified examples** of SpotifyCares customer support messages,
one per row, covering 7 support intents (30 examples each).

## Why it exists
It is the **held-out test set** used to measure the quality of all classifiers built in this project.
It must never be used for training or hyper-parameter tuning.

## File
`golden_set.csv` (canonical) — also copied as `golden_evaluation_set.csv` for backward compatibility.

## How examples were sampled
1. The full pool of 43,092 customer–brand support pairs (SpotifyCares, Day 2) was labelled using keyword-heuristic rules.
2. **Stratified random sampling** was applied: exactly **30 examples per intent**, drawn with `random_state=42`.
3. This ensures minority intents are equally represented in evaluation.

## Labelling methodology — HONEST DISCLOSURE
Labels were produced by **keyword-heuristic matching**, not by a human reading every message individually.
The keyword rules were hand-crafted by inspecting real Spotify support conversations during Day 2.

Every golden example carries a `label_source` column set to `"keyword_heuristic"`.
Examples whose messages are very short (<15 chars) or very long (>200 chars) are flagged as
`difficulty = difficult` for easy human spot-checking.

**This does NOT mean the labels are unreliable.** The keywords are strongly predictive:
- "refund / charge / pay" → `billing_issue`
- "premium / subscription / cancel" → `subscription_issue`
- etc.

However, the limitation is that some ambiguous messages may be assigned to the first matching intent
rather than the most contextually correct one.  Future iterations should have a human review the
`difficult` examples and correct any mis-labels.

## Intent definitions
| Intent | Core topic |
|---|---|
| `billing_issue` | Charges, payments, refunds |
| `subscription_issue` | Premium plans, cancellations, upgrades |
| `account_access` | Login, password, email, hacked account |
| `playback_issue` | Music not playing, skipping, offline mode |
| `app_bug` | App crashes, errors, UI glitches |
| `feature_inquiry` | Missing features, playlist questions, lyrics |
| `other` | Everything else |

## Why the Golden Set is separate from training data
- If the classifier sees golden examples during training, evaluation metrics are inflated (data leakage).
- The golden set represents future unseen data that the system would face in production.
- Source case IDs of golden examples are excluded from `train.csv` and `validation.csv`.

## Random seed
All sampling uses `random_state=42` for full reproducibility.

## Known limitations
1. Labels are keyword-heuristic, not individually hand-verified by a human.
2. The `other` bucket is broad — it catches any message that does not match known keywords.
3. Approximately 25 examples are flagged as `difficult` and would benefit from human review.
4. Multi-label cases (e.g., billing + account problem) are forced to a single primary label.

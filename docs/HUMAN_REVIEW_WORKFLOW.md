# Human Review Workflow — Golden Evaluation Set

## Purpose

The Golden Evaluation Set is the **held-out test set** for this project. Every label
must be trustworthy because downstream metrics (accuracy, F1, confusion matrix) are
computed directly from it.

This document defines the workflow a human reviewer must follow to convert the
keyword-heuristic labels into verified, hand-labelled ground truth.

## Files

| File | Role |
|---|---|
| `data/golden/golden_set.csv` | Canonical golden set (210 rows) |
| `data/golden/golden_evaluation_set.csv` | Legacy copy of the canonical set |
| `data/golden/human_review_queue.csv` | Subset of `golden_set.csv` that still needs human verification |
| `data/golden/golden_set_distribution.csv` | Intent counts / percentages |
| `data/golden/golden_set_quality.csv` | Quality metrics |

## How labels were created (honest disclosure)

The initial labels in `golden_set.csv` were produced by **keyword-heuristic
matching**, not by a human reading every message. The keyword rules were hand-crafted
by inspecting real Spotify support conversations during Day 2.

Every example carries `label_source = "keyword_heuristic"`.

This is documented honestly in `README.md` and `DECISIONS.md`. We do **not** claim
these labels are hand-labelled until a human has verified them.

## What the human reviewer must do

### Step 1 — Work through the review queue

Open `data/golden/human_review_queue.csv`. It contains every example flagged
`difficulty = "difficult"` (very short < 15 chars, or very long > 200 chars).

For each row:

1. Read `customer_message`.
2. Decide the **single most correct** intent from the 7 allowed labels:
   - `billing_issue`
   - `subscription_issue`
   - `account_access`
   - `playback_issue`
   - `app_bug`
   - `feature_inquiry`
   - `other`
3. If the keyword label is correct, leave `review_status = "verified_correct"`.
4. If the keyword label is wrong, change `review_status = "corrected"` and update
   the `intent` column with the correct label.
5. Optionally write a note in `reviewer_notes`.

### Step 2 — Spot-check the "normal" examples

The `normal` examples (190 of 210) were not flagged, but a reviewer should still
spot-check a random sample of at least 30 of them (roughly 15%) to confirm the
keyword rules are not systematically mis-labelling a common pattern.

If any `normal` example is found to be wrong, correct it in `golden_set.csv`,
set its `label_source` to `"human_corrected"`, and add a note.

### Step 3 — Re-label the whole set

After review, every example should have one of these `label_source` values:

| label_source | Meaning |
|---|---|
| `keyword_heuristic` | Label came from keyword rules and was **verified correct** by a human. |
| `human_corrected` | Label was **changed** by a human reviewer. |
| `keyword_heuristic_unverified` | (Optional) Not yet reviewed — do not leave in this state for long. |

### Step 4 — Update the golden set

After review is complete:

1. Run `python src/create_golden_set.py` to regenerate all derived files
   (distribution, quality, review queue) from the updated `golden_set.csv`.
2. Re-run `python src/check_data_leakage.py` to rebuild the splits.
3. Re-run `python src/evaluate_baselines.py` to recompute all metrics.
4. Re-run `python -m unittest discover -s tests -p "test_*.py" -v`.

### Step 5 — Update documentation

Update `README.md` and `DECISIONS.md` to reflect the actual labelling method used.
If any examples were corrected, report the number of corrections.

## Allowed labels

Only these 7 intents are valid. See `data/intent_definitions.csv` for the official
descriptions.

| Intent | Core topic |
|---|---|
| `billing_issue` | Charges, payments, refunds |
| `subscription_issue` | Premium plans, cancellations, upgrades |
| `account_access` | Login, password, email, hacked account |
| `playback_issue` | Music not playing, skipping, offline mode |
| `app_bug` | App crashes, errors, UI glitches |
| `feature_inquiry` | Missing features, playlist questions, lyrics |
| `other` | Everything else |

## What NOT to do

- Do **not** fabricate examples to reach a target count.
- Do **not** duplicate existing examples.
- Do **not** remove difficult examples to improve model performance.
- Do **not** change labels to make the model look better.
- Do **not** touch `train.csv` or `validation.csv` based on golden-set review —
  the golden set must remain evaluation-only.
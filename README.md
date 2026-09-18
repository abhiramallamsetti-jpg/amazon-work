# Hiver SDE Intern Take-Home Assignment

A customer-support AI project built on the Twitter customer support dataset, focused on intent classification, grounded response generation, and escalation logic for a selected brand.

## Overview

This project explores how to build a practical AI support agent for a real-world customer support workflow. The system is designed to:

- classify incoming customer messages by intent,
- retrieve historically relevant support patterns,
- generate grounded replies based on brand history,
- decide when to escalate to a human or a safe fallback,
- evaluate performance with baselines and an LLM-based judgment framework.

The selected brand for this work is SpotifyCares, which offers a manageable and realistic support dataset for local experimentation and prototype development.

## Problem Statement

The goal is to create an AI support assistant that can understand customer issues on social media and respond in a helpful, ground-safe, and brand-appropriate way.

A good response should:

- match the user intent correctly,
- be grounded in actual historical support data or known resolution patterns,
- avoid unsafe or unsupported account actions,
- escalate when confidence is low or the action is outside the system's authority.

## Dataset

The project uses the Customer Support on Twitter dataset from Kaggle.

- Original dataset size: 2,811,774 rows
- Original columns: `tweet_id`, `author_id`, `inbound`, `created_at`, `text`, `response_tweet_id`, `in_response_to_tweet_id`
- Brands covered: 108 unique brands
- Selected brand: `SpotifyCares`
- Reconstructed support pairs: 43,092 customer-brand pairs
- Intents identified: 7

### Intent Labels

- `billing_issue`
- `subscription_issue`
- `account_access`
- `playback_issue`
- `app_bug`
- `feature_inquiry`
- `other`

## Project Workflow

The pipeline is structured in stages:

1. Data inspection and brand filtering
2. Conversation reconstruction from customer-brand reply pairs
3. Support case filtering and intent discovery
4. Golden dataset creation and leakage checks
5. Baseline evaluation
6. Retrieval and response generation workflow
7. LLM judge and human review setup

## Baselines and Evaluation

The project includes a simple but important evaluation setup:

- Majority baseline: predicts the dominant class (`other`) for every sample
- TF-IDF + Logistic Regression baseline: a lightweight yet effective text classifier
- Quality checks for leakage, duplicate issues, and label integrity
- Golden evaluation set: 210 examples, evenly distributed across 7 intents

The baseline results show that the model performs meaningfully better than a trivial majority classifier, while also exposing the challenge of the broad `other` category and the need for stronger semantic understanding.

## System Status

This project is a strong research and internship-style prototype. It includes:

- dataset preparation and filtering,
- intent analysis,
- baseline modeling,
- evaluation scripts,
- retrieval and response-generation components,
- judge infrastructure and fallback logic.

Current notes:

- LLM judging is set up but depends on a valid API configuration.
- Human review for generated responses is prepared but still pending review execution.
- The system is intentionally conservative and safe when external model access is unavailable.

## Repository Structure

```text
hiver-sde-assignment/
├── data/
│   ├── golden/
│   ├── processed/
│   ├── raw/
│   ├── splits/
│   └── intent_definitions.csv
├── docs/
│   └── HUMAN_REVIEW_WORKFLOW.md
├── evaluation/
│   ├── human_review/
│   ├── results/
│   ├── judge_rubric.json
│   └── run_judge.py
├── notebooks/
├── scripts/
├── src/
│   ├── prompts/
│   ├── agent.py
│   ├── baseline_majority.py
│   ├── baseline_tfidf.py
│   ├── classifier.py
│   ├── judge.py
│   ├── retrieval.py
│   ├── reply_generator.py
│   └── ...
├── tests/
├── DECISIONS.md
├── REPORT.md
├── README.md
├── requirements.txt
└── .gitignore
```

## Setup

Create a Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Pipeline

### Data preparation

```bash
python src/inspect_dataset.py
python src/analyze_brands.py
python src/filter_brand.py
python src/data_quality_check.py
python src/conversation_builder.py
python src/filter_support_cases.py
python src/intent_analysis.py
```

### Golden set and baseline evaluation

```bash
python src/create_golden_set.py
python src/check_data_leakage.py
python src/evaluate_baselines.py
python -m unittest discover -s tests -p "test_*.py" -v
```

### Judge evaluation

```bash
python -m evaluation.run_judge
```

If the judge is configured with a valid LLM key, it can evaluate reply quality using the rubric defined in the project.

## Key Learnings

- Real customer support data is noisy, informal, and highly variable.
- Intent classification is feasible with a lightweight baseline but still difficult for broad categories like `other`.
- Grounded response generation is much more reliable when guided by historical support patterns.
- A safe escalation policy is essential for customer-support AI systems.
- Evaluation quality depends on human-reviewed labels and reliable model access.

## Limitations

- Current system is single-turn and brand-specific.
- The model relies on lightweight keyword-driven features in the baseline stage.
- Human review and API-backed judge evaluation are still pending completion.
- The project is a prototype for experimentation and internship-style learning, not a production-ready commercial support system.

## Future Improvements

- improve the intent taxonomy,
- handle multi-issue and multi-turn support conversations,
- add stronger semantic classifiers,
- expand the human-labeled evaluation set,
- integrate production-grade LLM inference and scoring pipelines.

## Summary

This project demonstrates a practical end-to-end approach to building an AI-based customer support agent using publicly available social-support data. It combines data preparation, baseline evaluation, retrieval, grounded response generation, and safety-focused escalation strategy in a way that is suitable for an internship project and portfolio presentation.

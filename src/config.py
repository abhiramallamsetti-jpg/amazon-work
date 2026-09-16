"""
src/config.py

Centralised configuration for the Day 4 AI support agent.

All thresholds, paths, and provider settings live here so that no other
module hardcodes magic numbers.  Values can be overridden by environment
variables (see ``.env.example``).
"""

import os
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
GOLDEN_DIR = DATA_DIR / "golden"
EVAL_DIR = ROOT / "evaluation"
RESULTS_DIR = EVAL_DIR / "results"
SAMPLE_INPUTS_PATH = DATA_DIR / "sample_inputs.json"
AGENT_DEMO_PATH = EVAL_DIR / "day4_agent_demo.json"

# Reusable Day 3 artefacts
TRAIN_CSV = SPLITS_DIR / "train.csv"
GOLDEN_CSV = GOLDEN_DIR / "golden_set.csv"
SUPPORT_CASES_CSV = PROCESSED_DIR / "support_cases.csv"


# ---------------------------------------------------------------------
# Intent classifier configuration
# ---------------------------------------------------------------------
# Reuse the exact Day 3 TF-IDF + Logistic Regression pipeline so that
# Day 4 behaviour is identical to the measured Day 3 baseline.
CLASSIFIER_MAX_FEATURES = 5000
CLASSIFIER_STOP_WORDS = "english"
CLASSIFIER_RANDOM_STATE = 42
CLASSIFIER_MAX_ITER = 1000

# Confidence threshold: derived from the Day 3 confidence analysis.
# The 0.6-0.8 bin held 39 correct / 10 incorrect predictions and the
# 0.8-1.0 bin held 121 correct / 9 incorrect.  A threshold of 0.6 keeps
# most correct predictions while still catching the ~33 errors, which
# mostly sit in the 0.4-0.6 and 0.6-0.8 bins.
CLASSIFIER_CONFIDENCE_THRESHOLD = 0.6


# ---------------------------------------------------------------------
# Retrieval configuration
# ---------------------------------------------------------------------
RETRIEVAL_TOP_K = 3
RETRIEVAL_MIN_SIMILARITY = 0.15
# TF-IDF shared with the classifier for consistency.
RETRIEVAL_MAX_FEATURES = 5000
RETRIEVAL_STOP_WORDS = "english"


# ---------------------------------------------------------------------
# Escalation configuration
# ---------------------------------------------------------------------
# Conservative defaults.  The agent prefers escalating over inventing
# unsupported information.
ESCALATE_ON_LOW_CONFIDENCE = True
ESCALATE_ON_WEAK_RETRIEVAL = True
ESCALATE_ON_HIGH_RISK = True
ESCALATE_ON_MULTI_INTENT = True
ESCALATE_ON_GROUNDING_FAILURE = True
ESCALATE_ON_LLM_FAILURE = True

MULTI_INTENT_MIN_TOP_INTENTS = 2
MULTI_INTENT_PROBABILITY_GAP = 0.15


# ---------------------------------------------------------------------
# Grounding configuration
# ---------------------------------------------------------------------
GROUNDING_MIN_LENGTH = 10
GROUNDING_MAX_LENGTH = 2000
# If the LLM reply is shorter than this we treat it as a failed generation.
GROUNDING_EMPTY_REPLY_THRESHOLD = 5


# ---------------------------------------------------------------------
# LLM provider configuration
# ---------------------------------------------------------------------
# Read from environment variables only.  Never hardcode a key.
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "")
LLM_TIMEOUT_SECONDS = float(os.environ.get("LLM_TIMEOUT_SECONDS", "30"))
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.0"))
# Rate limiter: seconds between API calls. 5.0 suits Google AI Studio (15 RPM);
# use 22.0 for OpenAI free tier (3 RPM).
LLM_MIN_INTERVAL_SECONDS = float(os.environ.get("LLM_MIN_INTERVAL_SECONDS", "5.0"))


@dataclass
class AgentConfig:
    """All tunable knobs for the agent, in one place."""

    # Classifier
    classifier_max_features: int = CLASSIFIER_MAX_FEATURES
    classifier_stop_words: str = CLASSIFIER_STOP_WORDS
    classifier_random_state: int = CLASSIFIER_RANDOM_STATE
    classifier_max_iter: int = CLASSIFIER_MAX_ITER
    classifier_confidence_threshold: float = CLASSIFIER_CONFIDENCE_THRESHOLD

    # Retrieval
    retrieval_top_k: int = RETRIEVAL_TOP_K
    retrieval_min_similarity: float = RETRIEVAL_MIN_SIMILARITY
    retrieval_max_features: int = RETRIEVAL_MAX_FEATURES
    retrieval_stop_words: str = RETRIEVAL_STOP_WORDS

    # Escalation
    escalate_on_low_confidence: bool = ESCALATE_ON_LOW_CONFIDENCE
    escalate_on_weak_retrieval: bool = ESCALATE_ON_WEAK_RETRIEVAL
    escalate_on_high_risk: bool = ESCALATE_ON_HIGH_RISK
    escalate_on_multi_intent: bool = ESCALATE_ON_MULTI_INTENT
    escalate_on_grounding_failure: bool = ESCALATE_ON_GROUNDING_FAILURE
    escalate_on_llm_failure: bool = ESCALATE_ON_LLM_FAILURE
    multi_intent_min_top_intents: int = MULTI_INTENT_MIN_TOP_INTENTS
    multi_intent_probability_gap: float = MULTI_INTENT_PROBABILITY_GAP

    # Grounding
    grounding_min_length: int = GROUNDING_MIN_LENGTH
    grounding_empty_reply_threshold: int = GROUNDING_EMPTY_REPLY_THRESHOLD

    # LLM
    llm_model: str = LLM_MODEL
    llm_timeout_seconds: float = LLM_TIMEOUT_SECONDS
    llm_temperature: float = LLM_TEMPERATURE
    llm_api_key: str = LLM_API_KEY
    llm_base_url: str = LLM_BASE_URL
    llm_min_interval_seconds: float = LLM_MIN_INTERVAL_SECONDS


# A single shared config instance used across the agent.
CONFIG = AgentConfig()
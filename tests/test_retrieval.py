"""
tests/test_retrieval.py

Unit tests for the Day 4 historical retrieval module.

These tests run without any LLM API and do not depend on external services.
"""

import os
import sys
import unittest

import pandas as pd

# Make the project root importable when running via unittest discover.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.retrieval import (
    HistoricalRetriever,
    _load_training_cases,
    build_retrieval_index,
)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
GOLDEN_DIR = os.path.join(ROOT, "data", "golden")


def _small_corpus() -> pd.DataFrame:
    """A tiny deterministic corpus for offline tests."""
    return pd.DataFrame({
        "case_id": ["c1", "c2", "c3", "c4"],
        "customer_message": [
            "my order is late and I want a refund",
            "I cannot log in to my account",
            "the app keeps crashing on start",
            "how do I download music for offline listening",
        ],
        "brand_reply": [
            "Let us look into your order and refund.",
            "Please reset your password.",
            "We are sorry, let us send a crash log.",
            "Open settings and tap offline downloads.",
        ],
        "intent": ["billing_issue", "account_access", "app_bug", "playback_issue"],
        "conversation_id": ["v1", "v2", "v3", "v4"],
    })


class TestRetrievalCorpus(unittest.TestCase):

    def test_load_training_cases_excludes_golden(self):
        cases = _load_training_cases()
        golden = pd.read_csv(os.path.join(GOLDEN_DIR, "golden_set.csv"),
                             keep_default_na=False)
        golden_ids = set(golden["source_case_id"].dropna().astype(str))
        overlap = cases["case_id"].astype(str).isin(golden_ids).sum()
        self.assertEqual(overlap, 0,
                         "Training retrieval corpus contains Golden Set cases.")

    def test_load_training_cases_non_empty(self):
        cases = _load_training_cases()
        self.assertGreater(len(cases), 0)


class TestRetrievalIndex(unittest.TestCase):

    def test_build_index_has_required_fields(self):
        index = build_retrieval_index(_small_corpus())
        self.assertIn("vectorizer", index)
        self.assertIn("matrix", index)
        self.assertIn("cases", index)
        self.assertGreater(index["corpus_size"], 0)

    def test_matrix_shape_matches_corpus(self):
        corpus = _small_corpus()
        index = build_retrieval_index(corpus)
        n_rows = index["matrix"].shape[0]
        self.assertEqual(n_rows, len(corpus))


class TestHistoricalRetriever(unittest.TestCase):

    def setUp(self):
        self.index = build_retrieval_index(_small_corpus())
        self.retriever = HistoricalRetriever(
            index=self.index, top_k=3, min_similarity=0.1
        )

    def test_query_returns_at_most_top_k(self):
        result = self.retriever.retrieve("my order is late refund")
        self.assertLessEqual(result["n_results"], 3)

    def test_results_have_required_fields(self):
        result = self.retriever.retrieve("my order is late refund")
        for case in result["top_cases"]:
            for field in ("case_id", "customer_message", "brand_reply",
                          "intent", "similarity_score"):
                self.assertIn(field, case)

    def test_similarity_descending(self):
        result = self.retriever.retrieve("my order is late refund")
        scores = [c["similarity_score"] for c in result["top_cases"]]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_empty_query_safe(self):
        result = self.retriever.retrieve("")
        self.assertFalse(result["sufficient"])
        self.assertEqual(result["top_cases"], [])
        result2 = self.retriever.retrieve(None)
        self.assertEqual(result2["top_cases"], [])

    def test_deterministic(self):
        q = "I cannot log in to my account"
        r1 = self.retriever.retrieve(q)
        r2 = self.retriever.retrieve(q)
        self.assertEqual(r1["top_cases"], r2["top_cases"])

    def test_min_similarity_threshold(self):
        retriever = HistoricalRetriever(
            index=self.index, top_k=3, min_similarity=0.99
        )
        result = retriever.retrieve("my order is late refund")
        self.assertFalse(result["sufficient"])

    def test_top_k_configurable(self):
        retriever_k2 = HistoricalRetriever(index=self.index, top_k=2)
        retriever_k4 = HistoricalRetriever(index=self.index, top_k=4)
        r2 = retriever_k2.retrieve("my order is late refund")
        r4 = retriever_k4.retrieve("my order is late refund")
        self.assertLessEqual(len(r2["top_cases"]), 2)
        self.assertLessEqual(len(r4["top_cases"]), 4)

    def test_relevant_case_is_top_result(self):
        result = self.retriever.retrieve("I cannot log in to my account")
        self.assertGreater(len(result["top_cases"]), 0)
        top = result["top_cases"][0]
        self.assertEqual(top["intent"], "account_access")


class TestRetrievalDeterminism(unittest.TestCase):

    def test_same_query_same_scores(self):
        index = build_retrieval_index(_small_corpus())
        r = HistoricalRetriever(index=index, top_k=3)
        a = r.retrieve("offline download music")
        b = r.retrieve("offline download music")
        self.assertEqual(
            [c["similarity_score"] for c in a["top_cases"]],
            [c["similarity_score"] for c in b["top_cases"]],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
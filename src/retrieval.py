
"""
src/retrieval.py

Historical support-case retrieval for the Day 4 AI support agent.

Retrieval method: TF-IDF + cosine similarity over historical customer
messages, with the corresponding brand replies returned as grounding
evidence.

DATA LEAKAGE RULE:
Golden Test cases are NEVER used as retrieval evidence.  The retrieval
corpus is built from the training split only, and any training row whose
``case_id`` matches a Golden Set ``source_case_id`` is explicitly excluded.
See ``DECISIONS.md`` Day 4 decision 3.

CASE DEDUPLICATION RULE:
One customer tweet can have multiple brand replies (the brand may answer
several times).  We group by ``customer_tweet_id`` so that each customer
case appears at most ONCE in the TOP-K results, while ALL associated brand
replies are preserved in the ``brand_replies`` list.  This prevents the same
customer message from occupying multiple TOP-K positions.
"""

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import (
    CONFIG,
    SPLITS_DIR,
    GOLDEN_DIR,
    SUPPORT_CASES_CSV,
)

INDEX_PATH = Path(__file__).resolve().parent.parent / "models" / "retrieval_index.pkl"


def _load_training_cases() -> pd.DataFrame:
    """Load training cases, excluding any that belong to the Golden Set."""
    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    golden_df = pd.read_csv(GOLDEN_DIR / "golden_set.csv", keep_default_na=False)

    # Exclude Golden Set source cases by case_id.
    golden_case_ids = set(golden_df["source_case_id"].dropna().astype(str))
    mask = ~train_df["case_id"].astype(str).isin(golden_case_ids)
    return train_df[mask].copy()


def _group_by_customer_tweet(cases: pd.DataFrame) -> pd.DataFrame:
    """
    Group rows so each customer case appears exactly once.

    The grouping key is ``customer_tweet_id`` when available, otherwise we
    fall back to ``case_id``.  The first row's scalar fields are kept
    (customer_message, intent, conversation_id, case_id), while ALL brand
    replies are aggregated into a list under ``brand_replies``.  This preserves
    useful historical resolution evidence instead of discarding all but the
    first reply.
    """
    cases = cases.copy()
    if "customer_tweet_id" not in cases.columns:
        # Fall back to case_id as the grouping key.
        cases["customer_tweet_id"] = cases["case_id"].astype(str)
    # Rename the scalar brand_reply column to brand_replies so the agg
    # dictionary can reference it.  We keep the original column too so the
    # aggregation can collect every reply into a list.
    if "brand_replies" not in cases.columns:
        cases["brand_replies"] = cases["brand_reply"]

    # Build the aggregation dict from only the columns that exist so the
    # function works with both the full training corpus and minimal test
    # corpora that may omit optional columns such as ``customer_user``.
    agg_dict = {
        "customer_message": "first",
        "case_id": "first",
        "conversation_id": "first",
        "intent": "first",
        "brand_replies": lambda s: list(s),
    }
    if "customer_user" in cases.columns:
        agg_dict["customer_user"] = "first"

    grouped = (
        cases.groupby("customer_tweet_id", sort=False)
              .agg(agg_dict)
              .reset_index()
    )
    # Drop the scalar brand_reply column if it survived the aggregation.
    if "brand_reply" in grouped.columns and "brand_replies" in grouped.columns:
        grouped = grouped.drop(columns=["brand_reply"])
    # Ensure every brand_replies entry is a list of non-empty strings.
    grouped["brand_replies"] = grouped["brand_replies"].apply(
        lambda v: [str(x) for x in v] if isinstance(v, list) else [str(v)]
    )
    return grouped


def build_retrieval_index(cases: pd.DataFrame | None = None) -> dict:
    """
    Build a TF-IDF retrieval index over historical customer messages.

    The corpus is deduplicated by customer tweet so each case is unique.
    Returns a dict containing the vectorizer, matrix, and case metadata.
    """
    if cases is None:
        cases = _load_training_cases()

    cases = cases.copy()
    cases["customer_message"] = cases["customer_message"].fillna("").astype(str)
    cases["brand_reply"] = cases["brand_reply"].fillna("").astype(str)
    cases["intent"] = cases["intent"].fillna("other").astype(str)
    cases["case_id"] = cases["case_id"].astype(str)
    cases["conversation_id"] = cases["conversation_id"].astype(str)

    # Deduplicate: one customer case, multiple brand replies preserved.
    cases = _group_by_customer_tweet(cases)

    vectorizer = TfidfVectorizer(
        max_features=CONFIG.retrieval_max_features,
        stop_words=CONFIG.retrieval_stop_words,
    )
    matrix = vectorizer.fit_transform(cases["customer_message"])

    return {
        "vectorizer": vectorizer,
        "matrix": matrix,
        "cases": cases,
        "corpus_size": int(len(cases)),
    }


def save_index(index: dict, path: Path | None = None) -> Path:
    if path is None:
        path = INDEX_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(index, f)
    return path


def _index_has_brand_replies(index: dict) -> bool:
    """Return True if the index DataFrame exposes a brand_replies column."""
    try:
        cases = index.get("cases")
        if cases is None:
            return False
        if hasattr(cases, "columns"):
            return "brand_replies" in cases.columns
    except Exception:
        return False
    return False


def load_index(path: Path | None = None) -> dict:
    if path is None:
        path = INDEX_PATH
    if path.exists():
        try:
            with open(path, "rb") as f:
                index = pickle.load(f)
            if _index_has_brand_replies(index):
                return index
        except Exception:
            pass
    # Rebuild when the pickle is missing, corrupt, or predates the schema change.
    index = build_retrieval_index()
    return save_index(index, path)


class HistoricalRetriever:
    """
    Retrieve the most similar historical support cases for a query.

    Parameters
    ----------
    index : dict, optional
        Pre-built retrieval index.  Built on demand if None.
    top_k : int, optional
        Maximum number of cases to return.
    min_similarity : float, optional
        Cases below this cosine similarity are not returned.
    """

    def __init__(self, index: dict | None = None,
                 top_k: int | None = None,
                 min_similarity: float | None = None):
        if index is None:
            index = load_index()
        self.index = index
        self.top_k = top_k if top_k is not None else CONFIG.retrieval_top_k
        self.min_similarity = (
            min_similarity if min_similarity is not None
            else CONFIG.retrieval_min_similarity
        )

    def retrieve(self, query: str) -> dict:
        """
        Retrieve top-k similar historical cases.

        Returns
        -------
        dict with keys:
            sufficient      -- bool, whether any case exceeded min_similarity
            top_cases       -- list of case dicts
            top_similarity  -- float, highest similarity found (0.0 if none)
            n_results       -- int
        """
        if query is None:
            query = ""
        query = str(query).strip()

        if not query:
            return {
                "sufficient": False,
                "top_cases": [],
                "top_similarity": 0.0,
                "n_results": 0,
            }

        vectorizer = self.index["vectorizer"]
        matrix = self.index["matrix"]
        cases = self.index["cases"]

        query_vec = vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, matrix)[0]

        # Sort by similarity descending and take the top-k UNIQUE cases.
        top_indices = np.argsort(similarities)[::-1][: self.top_k]

        retrieved = []
        top_sim = 0.0
        for idx in top_indices:
            sim = float(similarities[idx])
            if sim <= 0:
                continue
            row = cases.iloc[int(idx)]
            brand_replies = row.get("brand_replies", [])
            if isinstance(brand_replies, str):
                brand_replies = [brand_replies]
            retrieved.append({
                "case_id": str(row["case_id"]),
                "customer_tweet_id": str(row.get("customer_tweet_id", "")),
                "customer_message": str(row["customer_message"]),
                "brand_replies": brand_replies,
                "brand_reply": brand_replies[0] if brand_replies else "",
                "intent": str(row["intent"]),
                "similarity_score": round(sim, 4),
                "conversation_id": str(row.get("conversation_id", "")),
            })
            if sim > top_sim:
                top_sim = sim

        sufficient = top_sim >= self.min_similarity
        return {
            "sufficient": sufficient,
            "top_cases": retrieved,
            "top_similarity": round(top_sim, 4),
            "n_results": len(retrieved),
        }


def main():
    """Build and persist the retrieval index, printing a summary."""
    index = build_retrieval_index()
    path = save_index(index)
    retriever = HistoricalRetriever(index)
    result = retriever.retrieve("My order is late and I want a refund.")
    print(f"Index saved to {path}")
    print(f"Corpus size: {index['corpus_size']}")
    print(f"Retrieval sufficient: {result['sufficient']}")
    print(f"Top similarity: {result['top_similarity']}")
    for c in result["top_cases"]:
        print(f"  [{c['similarity_score']}] {c['intent']}: {c['customer_message'][:80]}")
        print(f"      brand_replies: {len(c['brand_replies'])}")


if __name__ == "__main__":
    main()

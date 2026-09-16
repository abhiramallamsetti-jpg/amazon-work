"""test_agent.py"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agent import SupportAgent, _validate_input
from src.classifier import IntentClassifier
from src.retrieval import HistoricalRetriever
from src.reply_generator import ReplyGenerator
from src.llm_provider import FallbackLLMProvider
from src.grounding import check_grounding
from src.escalation import decide_escalation


class MockLLMProvider(FallbackLLMProvider):
    def __init__(self, reply="Thank you. I will look into this."):
        super().__init__()
        self._reply = reply
        self.calls = 0

    def generate(self, prompt, **kwargs):
        self.calls += 1
        return self._reply


class TestInputValidation(unittest.TestCase):

    def test_empty_message_invalid(self):
        r = _validate_input("")
        self.assertFalse(r["valid"])

    def test_none_message_invalid(self):
        r = _validate_input(None)
        self.assertFalse(r["valid"])

    def test_whitespace_invalid(self):
        r = _validate_input("   ")
        self.assertFalse(r["valid"])

    def test_valid_message(self):
        r = _validate_input("Where is my order?")
        self.assertTrue(r["valid"])


class TestAgentOutputStructure(unittest.TestCase):

    def setUp(self):
        self.agent = SupportAgent(
            classifier=IntentClassifier(),
            retriever=HistoricalRetriever(),
            reply_generator=ReplyGenerator(provider=MockLLMProvider()),
        )

    def test_normal_input_returns_required_fields(self):
        result = self.agent.handle("Where is my order? It is late.")
        for field in ("customer_message", "intent", "intent_confidence",
                      "retrieval", "decision", "escalation_reason",
                      "reply", "grounding", "evidence"):
            self.assertIn(field, result, "Missing field: " + field)

    def test_empty_input_escalates(self):
        result = self.agent.handle("")
        self.assertEqual(result["decision"], "ESCALATE")

    def test_prediction_count_matches(self):
        result = self.agent.handle("Where is my order? It is late.")
        self.assertEqual(len(result["evidence"]),
                         len(result["retrieval"]["top_cases"]))

    def test_grounding_field_present(self):
        result = self.agent.handle("Where is my order? It is late.")
        self.assertIn("supported", result["grounding"])

    def test_high_risk_escalates(self):
        result = self.agent.handle(
            "I think my account has been hacked. Someone changed my email."
        )
        self.assertEqual(result["decision"], "ESCALATE")

    def test_account_action_escalates(self):
        result = self.agent.handle("Please change my delivery address.")
        self.assertEqual(result["decision"], "ESCALATE")

    def test_low_confidence_escalates(self):
        result = self.agent.handle("asdfqwerty zxcvbnm qwerty asdf")
        self.assertEqual(result["decision"], "ESCALATE")

    def test_multiple_intents_escalates(self):
        result = self.agent.handle(
            "My order is late, I was charged twice, and please change my address."
        )
        self.assertEqual(result["decision"], "ESCALATE")

    def test_mock_llm_used(self):
        provider = MockLLMProvider(reply="Draft reply here.")
        agent = SupportAgent(
            classifier=IntentClassifier(),
            retriever=HistoricalRetriever(),
            reply_generator=ReplyGenerator(provider=provider),
        )
        agent.handle("Where is my order?")
        self.assertGreater(provider.calls, 0)


class TestEscalationOverride(unittest.TestCase):

    def test_grounding_failure_forces_escalate(self):
        decision = decide_escalation(
            customer_message="Where is my order?",
            classification={"intent": "other", "confidence": 0.9,
                             "top_intents": [{"intent": "other", "probability": 0.9}]},
            retrieval={"sufficient": True, "top_similarity": 0.8,
                       "top_cases": [{"case_id": "c1"}]},
            grounding_result={"supported": False, "issues": ["bad"]},
        )
        self.assertEqual(decision["decision"], "ESCALATE")

    def test_llm_failure_forces_escalate(self):
        decision = decide_escalation(
            customer_message="Where is my order?",
            classification={"intent": "other", "confidence": 0.9,
                             "top_intents": [{"intent": "other", "probability": 0.9}]},
            retrieval={"sufficient": True, "top_similarity": 0.8,
                       "top_cases": [{"case_id": "c1"}]},
            llm_failure=True,
        )
        self.assertEqual(decision["decision"], "ESCALATE")


class TestGroundingChecks(unittest.TestCase):

    def test_empty_reply_unsupported(self):
        r = check_grounding("", {"top_cases": [{"case_id": "c1"}]},
                            {"decision": "AUTO_HANDLE"})
        self.assertFalse(r["supported"])

    def test_completed_action_unsupported(self):
        r = check_grounding(
            "Your order has been cancelled and you will receive a refund of $50.",
            {"top_cases": [{"case_id": "c1"}]},
            {"decision": "AUTO_HANDLE"},
        )
        self.assertFalse(r["supported"])
        self.assertGreater(len(r["completed_action_claims"]), 0)

    def test_clean_reply_supported(self):
        r = check_grounding(
            "I can help with that. Please try clearing the app cache.",
            {"top_cases": [{"case_id": "c1"}]},
            {"decision": "AUTO_HANDLE"},
        )
        self.assertTrue(r["supported"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
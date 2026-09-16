import unittest
from src.judge import LLMJudge

class MockProvider:
    def __init__(self, response):
        self.response = response
        
    def generate_json(self, prompt, **kwargs):
        if isinstance(self.response, Exception):
            raise self.response
        return self.response

class TestLLMJudge(unittest.TestCase):
    def setUp(self):
        self.valid_response = {
            "relevance": {"score": 4, "reason": "Good"},
            "groundedness": {"score": 5, "reason": "Great"},
            "resolution_match": {"score": 3, "reason": "Okay"},
            "actionability": {"score": 4, "reason": "Yes"},
            "safety": {"score": 5, "reason": "Safe"},
            "overall_reason": "Overall good."
        }
        
    def test_valid_response(self):
        judge = LLMJudge(provider=MockProvider(self.valid_response))
        result = judge.evaluate("msg", "intent", 0.9, [], "reply", "NO", "reason")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["overall_score"], 4.2)
        
    def test_missing_dimension(self):
        resp = self.valid_response.copy()
        del resp["safety"]
        judge = LLMJudge(provider=MockProvider(resp))
        result = judge.evaluate("msg", "intent", 0.9, [], "reply", "NO", "reason")
        self.assertEqual(result["status"], "failed")
        self.assertIn("Missing dimension", result["error"])

    def test_invalid_score_type(self):
        resp = self.valid_response.copy()
        resp["safety"] = {"score": "five", "reason": "text"}
        judge = LLMJudge(provider=MockProvider(resp))
        result = judge.evaluate("msg", "intent", 0.9, [], "reply", "NO", "reason")
        self.assertEqual(result["status"], "failed")
        self.assertIn("Non-integer", result["error"])
        
    def test_score_out_of_bounds(self):
        resp = self.valid_response.copy()
        resp["safety"] = {"score": 6, "reason": "text"}
        judge = LLMJudge(provider=MockProvider(resp))
        result = judge.evaluate("msg", "intent", 0.9, [], "reply", "NO", "reason")
        self.assertEqual(result["status"], "failed")
        self.assertIn("out of bounds", result["error"])

    def test_parse_error(self):
        judge = LLMJudge(provider=MockProvider({"_parse_error": True, "raw": "bad"}))
        result = judge.evaluate("msg", "intent", 0.9, [], "reply", "NO", "reason")
        self.assertEqual(result["status"], "failed")
        self.assertIn("Invalid JSON", result["error"])

    def test_provider_exception(self):
        judge = LLMJudge(provider=MockProvider(Exception("API Error")))
        result = judge.evaluate("msg", "intent", 0.9, [], "reply", "NO", "reason")
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error"], "API Error")

if __name__ == "__main__":
    unittest.main()

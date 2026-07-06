import unittest

from llm_eval.graders import GRADERS
from llm_eval.metrics import cost_usd


class TestGraders(unittest.TestCase):
    def test_exact_match_normalizes(self):
        self.assertTrue(GRADERS["exact_match"]("  Positive\n", "positive").passed)
        self.assertFalse(GRADERS["exact_match"]("negative", "positive").passed)

    def test_contains(self):
        self.assertTrue(GRADERS["contains"]("the answer is 42", "42").passed)
        self.assertFalse(GRADERS["contains"]("no digits here", "42").passed)

    def test_json_fields_subset_match(self):
        resp = 'Sure! {"seniority": "senior", "remote": true, "title": "X"}'
        self.assertTrue(GRADERS["json_fields"](resp, {"seniority": "senior", "remote": True}).passed)

    def test_json_fields_reports_mismatch(self):
        resp = '{"remote": false}'
        result = GRADERS["json_fields"](resp, {"remote": True})
        self.assertFalse(result.passed)
        self.assertIn("remote", result.detail)

    def test_json_fields_handles_garbage(self):
        self.assertFalse(GRADERS["json_fields"]("not json at all", {"a": 1}).passed)


class TestCost(unittest.TestCase):
    def test_opus_pricing(self):
        # 1M input + 1M output on opus-4-8 = $5 + $25
        self.assertAlmostEqual(cost_usd("claude-opus-4-8", 1_000_000, 1_000_000), 30.0)

    def test_local_model_is_free(self):
        self.assertIsNone(cost_usd("llama3.2", 1000, 1000))

    def test_prefix_match(self):
        # real API returns dated/suffixed model ids
        self.assertIsNotNone(cost_usd("claude-sonnet-5-20260101", 100, 100))


if __name__ == "__main__":
    unittest.main()

import asyncio
import unittest
from pathlib import Path

from llm_eval.graders.judge import llm_judge
from llm_eval.providers.base import Completion, Provider
from llm_eval.runner import run_eval

EVALS = Path(__file__).resolve().parent.parent / "evals"


class StubJudge(Provider):
    """A judge whose verdict is fixed — lets us test the grader path offline."""

    def __init__(self, text: str):
        self.text = text
        self.name = "stub-judge"

    async def complete(self, prompt: str, system: str = "") -> Completion:
        return Completion(self.text, "stub", 10, 5, 0.01)


class TestJudgeGrader(unittest.TestCase):
    def test_parses_pass(self):
        judge = StubJudge("Reasoning here.\nVERDICT: PASS")
        result = asyncio.run(llm_judge(judge, "q", "an answer", "rubric"))
        self.assertTrue(result.passed)
        self.assertIn("Reasoning", result.detail)

    def test_parses_fail(self):
        judge = StubJudge("The answer misses the point.\nVERDICT: FAIL")
        result = asyncio.run(llm_judge(judge, "q", "an answer", "rubric"))
        self.assertFalse(result.passed)

    def test_missing_verdict_is_failure(self):
        judge = StubJudge("I think it's fine but I won't say the magic word.")
        result = asyncio.run(llm_judge(judge, "q", "an answer", "rubric"))
        self.assertFalse(result.passed)
        self.assertIn("no VERDICT", result.detail)

    def test_empty_answer_shortcircuits(self):
        judge = StubJudge("VERDICT: PASS")  # would pass if called
        result = asyncio.run(llm_judge(judge, "q", "   ", "rubric"))
        self.assertFalse(result.passed)


class TestJudgeThroughRunner(unittest.TestCase):
    def test_injected_judge_drives_verdicts(self):
        run = asyncio.run(
            run_eval(EVALS / "judge_demo.yaml", ["mock"],
                     judge_provider=StubJudge("Looks correct.\nVERDICT: PASS"))
        )
        self.assertEqual(len(run.results), 2)
        self.assertTrue(all(r.passed for r in run.results))


if __name__ == "__main__":
    unittest.main()

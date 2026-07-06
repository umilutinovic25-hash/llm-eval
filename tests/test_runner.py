import asyncio
import unittest
from pathlib import Path

from llm_eval.runner import run_eval

EVALS = Path(__file__).resolve().parent.parent / "evals"


class TestRunner(unittest.TestCase):
    def test_smoke_eval_all_pass_with_mock(self):
        run = asyncio.run(run_eval(EVALS / "smoke.yaml", ["mock"]))
        self.assertEqual(len(run.results), 2)
        self.assertTrue(all(r.passed for r in run.results))
        self.assertTrue(all(r.error is None for r in run.results))

    def test_multiple_providers_produce_rows_each(self):
        run = asyncio.run(run_eval(EVALS / "smoke.yaml", ["mock", "mock"]))
        self.assertEqual(len(run.results), 4)


if __name__ == "__main__":
    unittest.main()

import json
import tempfile
import unittest
from pathlib import Path

from llm_eval.html import latest_per_eval, load_snapshot, render_dashboard
from llm_eval.runner import CaseResult, RunResult


def _run(name, started, rows):
    results = [
        CaseResult(
            case_id=f"c{i}", provider=prov, passed=passed, detail="", response="",
            input_tokens=10, output_tokens=5, latency_s=lat, cost_usd=cost,
        )
        for prov, passed, lat, cost in rows
        for i in [rows.index((prov, passed, lat, cost))]
    ]
    return RunResult(eval_name=name, started_at=started, results=results)


class TestHtml(unittest.TestCase):
    def test_dashboard_contains_eval_and_provider(self):
        run = _run("myeval", "2026-07-06T10:00:00",
                   [("ollama:llama3.2", True, 1.0, None), ("claude:opus", False, 0.5, 0.001)])
        out = render_dashboard([run])
        self.assertIn("myeval", out)
        self.assertIn("ollama:llama3.2", out)
        self.assertIn("<!doctype html>", out.lower())
        self.assertIn("free", out)       # local model → free
        self.assertIn("$0.0010", out)    # claude cost formatted

    def test_snapshot_roundtrip(self):
        run = _run("e", "2026-07-06T10:00:00", [("mock", True, 0.1, None)])
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "snap.json"
            p.write_text(run.to_json())
            loaded = load_snapshot(p)
            self.assertEqual(loaded.eval_name, "e")
            self.assertEqual(len(loaded.results), 1)
            self.assertTrue(loaded.results[0].passed)

    def test_latest_per_eval_keeps_newest(self):
        with tempfile.TemporaryDirectory() as d:
            old = Path(d) / "old.json"
            new = Path(d) / "new.json"
            old.write_text(_run("x", "2026-07-06T10:00:00", [("mock", False, 0.1, None)]).to_json())
            new.write_text(_run("x", "2026-07-06T12:00:00", [("mock", True, 0.1, None)]).to_json())
            runs = latest_per_eval([old, new])
            self.assertEqual(len(runs), 1)
            self.assertEqual(runs[0].started_at, "2026-07-06T12:00:00")


if __name__ == "__main__":
    unittest.main()

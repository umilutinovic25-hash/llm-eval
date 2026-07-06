from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from rich.console import Console
from rich.table import Table

from llm_eval.metrics import summarize
from llm_eval.runner import CaseResult, RunResult


def print_report(run: RunResult) -> None:
    console = Console()
    by_provider: Dict[str, List[CaseResult]] = defaultdict(list)
    for r in run.results:
        by_provider[r.provider].append(r)

    table = Table(title=f"Eval: {run.eval_name}  ({run.started_at})")
    table.add_column("Provider", style="bold")
    table.add_column("Pass rate", justify="right")
    table.add_column("Errors", justify="right")
    table.add_column("Latency p50", justify="right")
    table.add_column("Latency p95", justify="right")
    table.add_column("Cost (run)", justify="right")

    for provider, results in sorted(by_provider.items()):
        graded = [r for r in results if r.error is None]
        passed = sum(1 for r in graded if r.passed)
        errors = len(results) - len(graded)
        lat = summarize([r.latency_s for r in graded])
        costs = [r.cost_usd for r in graded if r.cost_usd is not None]
        cost_str = f"${sum(costs):.4f}" if costs else "free"
        rate = f"{passed}/{len(graded)}" if graded else "—"
        style = "green" if graded and passed == len(graded) else ("red" if errors else "")
        table.add_row(
            provider, rate, str(errors),
            f"{lat['p50']:.2f}s", f"{lat['p95']:.2f}s", cost_str,
            style=style or None,
        )
    console.print(table)

    failures = [r for r in run.results if not r.passed]
    if failures:
        console.print("\n[bold]Failures:[/bold]")
        for r in failures:
            reason = r.error or r.detail or "failed"
            console.print(f"  [red]✗[/red] {r.provider} / {r.case_id}: {reason}")

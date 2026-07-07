"""Render one or more eval snapshots into a single self-contained HTML dashboard.

No external assets — the CSS is inlined so the file opens anywhere and can be
committed or emailed as-is. Theme-aware (light/dark via prefers-color-scheme).
"""
from __future__ import annotations

import html as _html
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from llm_eval.metrics import summarize
from llm_eval.runner import CaseResult, RunResult


def load_snapshot(path: Path) -> RunResult:
    data = json.loads(Path(path).read_text())
    results = [CaseResult(**c) for c in data["results"]]
    return RunResult(eval_name=data["eval_name"], started_at=data["started_at"], results=results)


def latest_per_eval(paths: List[Path]) -> List[RunResult]:
    """Load snapshots, keeping only the most recent run for each eval name."""
    best: Dict[str, RunResult] = {}
    for p in paths:
        run = load_snapshot(p)
        if run.eval_name not in best or run.started_at > best[run.eval_name].started_at:
            best[run.eval_name] = run
    return sorted(best.values(), key=lambda r: r.eval_name)


def _provider_rows(run: RunResult):
    by_provider: Dict[str, List[CaseResult]] = defaultdict(list)
    for r in run.results:
        by_provider[r.provider].append(r)
    rows = []
    for provider, results in sorted(by_provider.items()):
        graded = [r for r in results if r.error is None]
        passed = sum(1 for r in graded if r.passed)
        total = len(graded)
        lat = summarize([r.latency_s for r in graded])
        costs = [r.cost_usd for r in graded if r.cost_usd is not None]
        rows.append({
            "provider": provider,
            "passed": passed,
            "total": total,
            "rate": (passed / total) if total else 0.0,
            "errors": len(results) - total,
            "p50": lat["p50"],
            "p95": lat["p95"],
            "cost": sum(costs) if costs else None,
        })
    return rows


def _rate_class(rate: float) -> str:
    if rate >= 0.999:
        return "full"
    if rate >= 0.5:
        return "mid"
    return "low"


def _card(run: RunResult) -> str:
    rows_html = []
    for row in _provider_rows(run):
        pct = round(row["rate"] * 100)
        cost = "free" if row["cost"] is None else f"${row['cost']:.4f}"
        err = f'<span class="err">{row["errors"]}</span>' if row["errors"] else "0"
        rows_html.append(f"""
        <tr>
          <td class="prov">{_html.escape(row["provider"])}</td>
          <td class="rate">
            <div class="bar"><div class="fill {_rate_class(row['rate'])}" style="width:{pct}%"></div></div>
            <span class="frac">{row["passed"]}/{row["total"]}</span>
          </td>
          <td class="num">{err}</td>
          <td class="num">{row["p50"]:.2f}s</td>
          <td class="num">{row["p95"]:.2f}s</td>
          <td class="num">{cost}</td>
        </tr>""")
    return f"""
    <section class="card">
      <div class="card-head">
        <h2>{_html.escape(run.eval_name)}</h2>
        <span class="stamp">{_html.escape(run.started_at)}</span>
      </div>
      <div class="table-scroll">
        <table>
          <thead><tr>
            <th>Provider</th><th>Pass rate</th><th>Err</th>
            <th>p50</th><th>p95</th><th>Cost</th>
          </tr></thead>
          <tbody>{''.join(rows_html)}</tbody>
        </table>
      </div>
    </section>"""


def render_dashboard(runs: List[RunResult]) -> str:
    cards = "\n".join(_card(r) for r in runs)
    eval_count = len(runs)
    providers = sorted({r.provider for run in runs for r in run.results})
    subtitle = f"{eval_count} eval{'s' if eval_count != 1 else ''} · {len(providers)} providers"
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>llm-eval report</title>
<style>
  :root {{
    --bg: #f7f7f5; --card: #ffffff; --ink: #1a1a1a; --muted: #6b6b6b;
    --line: #e5e5e0; --track: #ececea;
    --full: #2f9e5f; --mid: #d9a441; --low: #d1584e; --accent: #3b5bdb;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --bg: #14151a; --card: #1d1f27; --ink: #eceef2; --muted: #9aa0ab;
      --line: #2c2f3a; --track: #2a2d37;
      --full: #46c07a; --mid: #e0b354; --low: #e06b60; --accent: #6b83f0;
    }}
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 2.5rem 1.25rem 4rem; background: var(--bg); color: var(--ink);
    font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  }}
  .wrap {{ max-width: 860px; margin: 0 auto; }}
  header {{ margin-bottom: 2rem; }}
  h1 {{ margin: 0 0 .25rem; font-size: 1.7rem; letter-spacing: -.02em; }}
  .sub {{ color: var(--muted); font-size: .95rem; }}
  .card {{
    background: var(--card); border: 1px solid var(--line); border-radius: 12px;
    padding: 1.1rem 1.25rem 1.25rem; margin-bottom: 1.1rem;
  }}
  .card-head {{ display: flex; align-items: baseline; justify-content: space-between; margin-bottom: .6rem; }}
  .card-head h2 {{ margin: 0; font-size: 1.15rem; letter-spacing: -.01em; }}
  .stamp {{ color: var(--muted); font-size: .8rem; font-variant-numeric: tabular-nums; }}
  .table-scroll {{ overflow-x: auto; }}
  table {{ width: 100%; border-collapse: collapse; }}
  th, td {{ text-align: left; padding: .5rem .5rem; border-bottom: 1px solid var(--line); }}
  th {{ font-size: .72rem; text-transform: uppercase; letter-spacing: .05em; color: var(--muted); font-weight: 600; }}
  tbody tr:last-child td {{ border-bottom: none; }}
  .prov {{ font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: .85rem; white-space: nowrap; }}
  .num {{ text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }}
  .rate {{ display: flex; align-items: center; gap: .6rem; min-width: 160px; }}
  .bar {{ flex: 1; height: 8px; background: var(--track); border-radius: 999px; overflow: hidden; }}
  .fill {{ height: 100%; border-radius: 999px; }}
  .fill.full {{ background: var(--full); }}
  .fill.mid {{ background: var(--mid); }}
  .fill.low {{ background: var(--low); }}
  .frac {{ font-variant-numeric: tabular-nums; font-size: .85rem; color: var(--muted); min-width: 34px; }}
  .err {{ color: var(--low); font-weight: 600; }}
  footer {{ margin-top: 2rem; color: var(--muted); font-size: .8rem; text-align: center; }}
</style>
</head>
<body>
  <div class="wrap">
    <header>
      <h1>llm-eval report</h1>
      <div class="sub">{_html.escape(subtitle)}</div>
    </header>
    {cards}
    <footer>Generated by llm-eval · pass rate = graded cases passed; errors excluded from rate</footer>
  </div>
</body>
</html>"""


def render_to_file(runs: List[RunResult], out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.write_text(render_dashboard(runs))
    return out_path

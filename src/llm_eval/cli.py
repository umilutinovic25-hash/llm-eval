from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from llm_eval.report import print_report
from llm_eval.runner import run_eval, save_run


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="llm-eval",
        description="Run an eval across one or more LLM providers and compare results.",
    )
    parser.add_argument("eval_file", type=Path, help="Path to an eval YAML file")
    parser.add_argument(
        "-p", "--provider", action="append", required=True, dest="providers",
        help="Provider spec: mock, ollama:<model>, claude:<model>. Repeatable.",
    )
    parser.add_argument("-c", "--concurrency", type=int, default=4)
    parser.add_argument(
        "--results-dir", type=Path, default=Path("results"),
        help="Where to store the JSON snapshot of this run",
    )
    args = parser.parse_args()

    if not args.eval_file.exists():
        sys.exit(f"Eval file not found: {args.eval_file}")

    run = asyncio.run(run_eval(args.eval_file, args.providers, args.concurrency))
    print_report(run)
    out = save_run(run, args.results_dir)
    print(f"\nSnapshot: {out}")

    if any(not r.passed for r in run.results):
        sys.exit(1)


if __name__ == "__main__":
    main()

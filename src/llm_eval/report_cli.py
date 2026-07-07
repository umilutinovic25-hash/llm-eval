from __future__ import annotations

import argparse
import sys
from pathlib import Path

from llm_eval.html import latest_per_eval, render_to_file


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="llm-eval-report",
        description="Aggregate eval JSON snapshots into one self-contained HTML dashboard.",
    )
    parser.add_argument(
        "snapshots", nargs="*", type=Path,
        help="Snapshot JSON files. If omitted, uses the latest per eval in --results-dir.",
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("-o", "--out", type=Path, default=Path("report.html"))
    args = parser.parse_args()

    paths = args.snapshots or sorted(args.results_dir.glob("*.json"))
    if not paths:
        sys.exit(f"No snapshots found (looked in {args.results_dir}). Run an eval first.")

    runs = latest_per_eval(paths)
    out = render_to_file(runs, args.out)
    print(f"Wrote {out} ({len(runs)} evals)")


if __name__ == "__main__":
    main()

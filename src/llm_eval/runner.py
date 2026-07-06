from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from llm_eval.graders import GRADERS
from llm_eval.graders.judge import llm_judge
from llm_eval.metrics import cost_usd
from llm_eval.providers import Provider, create_provider

JUDGE_GRADER = "llm_judge"


@dataclass
class EvalCase:
    id: str
    prompt: str
    expected: Any
    grader: Optional[str] = None  # falls back to the eval-level grader


@dataclass
class EvalSpec:
    name: str
    grader: str
    cases: List[EvalCase]
    description: str = ""
    system: str = ""
    judge: str = "claude:claude-opus-4-8"  # provider used when grader is llm_judge

    @classmethod
    def load(cls, path: Path) -> "EvalSpec":
        raw = yaml.safe_load(path.read_text())
        cases = [EvalCase(**c) for c in raw.pop("cases")]
        return cls(cases=cases, **raw)


@dataclass
class CaseResult:
    case_id: str
    provider: str
    passed: bool
    detail: str
    response: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    cost_usd: Optional[float]
    error: Optional[str] = None


@dataclass
class RunResult:
    eval_name: str
    started_at: str
    results: List[CaseResult] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)


async def _run_case(
    provider: Provider,
    case: EvalCase,
    spec: EvalSpec,
    sem: asyncio.Semaphore,
    judge: Optional[Provider] = None,
) -> CaseResult:
    grader_name = case.grader or spec.grader
    async with sem:
        try:
            completion = await provider.complete(case.prompt, system=spec.system)
            if grader_name == JUDGE_GRADER:
                grade = await llm_judge(judge, case.prompt, completion.text, str(case.expected))
            else:
                grade = GRADERS[grader_name](completion.text, case.expected)
        except Exception as e:
            return CaseResult(
                case_id=case.id, provider=provider.name, passed=False,
                detail="", response="", input_tokens=0, output_tokens=0,
                latency_s=0.0, cost_usd=None, error=f"{type(e).__name__}: {e}",
            )
    return CaseResult(
        case_id=case.id,
        provider=provider.name,
        passed=grade.passed,
        detail=grade.detail,
        response=completion.text,
        input_tokens=completion.input_tokens,
        output_tokens=completion.output_tokens,
        latency_s=completion.latency_s,
        cost_usd=cost_usd(completion.model, completion.input_tokens, completion.output_tokens),
    )


async def run_eval(
    spec_path: Path,
    provider_specs: List[str],
    concurrency: int = 4,
    judge_provider: Optional[Provider] = None,
) -> RunResult:
    spec = EvalSpec.load(spec_path)
    used_graders = {c.grader or spec.grader for c in spec.cases}
    unknown = used_graders - set(GRADERS) - {JUDGE_GRADER}
    if unknown:
        raise ValueError(f"Unknown grader(s): {sorted(unknown)}")

    # Only spin up a judge if some case needs one; allow injecting one (tests, shared client).
    judge = judge_provider
    if JUDGE_GRADER in used_graders and judge is None:
        judge = create_provider(spec.judge)

    providers = [create_provider(s) for s in provider_specs]
    sem = asyncio.Semaphore(concurrency)
    tasks = [
        _run_case(provider, case, spec, sem, judge)
        for provider in providers
        for case in spec.cases
    ]
    run = RunResult(
        eval_name=spec.name,
        started_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        results=list(await asyncio.gather(*tasks)),
    )
    return run


def save_run(run: RunResult, results_dir: Path) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = run.started_at.replace(":", "-")
    out = results_dir / f"{run.eval_name}_{stamp}.json"
    out.write_text(run.to_json())
    return out

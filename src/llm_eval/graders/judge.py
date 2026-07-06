"""LLM-as-judge grader for open-ended answers.

Unlike the deterministic graders in __init__.py, this one is async and needs a
*judge* provider (any Provider — Claude, Ollama, or a stub in tests). The judge
reads the question, the candidate answer, and a rubric, then returns PASS/FAIL
with a one-line reason.

Design notes for the reader:
  * The judge is provider-agnostic — you can judge cheap-model output with an
    expensive judge, or self-judge with the same model.
  * The verdict is parsed from a strict `VERDICT: PASS|FAIL` line. Asking the
    judge to reason *before* committing to a verdict reduces leniency bias
    compared to a bare yes/no.
  * Judge calls cost tokens too; that cost is intentionally not folded into the
    model-under-test's cost column so the comparison stays honest.
"""
from __future__ import annotations

import re

from llm_eval.graders import GradeResult
from llm_eval.providers.base import Provider

JUDGE_SYSTEM = (
    "You are a strict, fair grader. You are given a QUESTION, a candidate "
    "ANSWER, and a RUBRIC describing what a correct answer must satisfy. "
    "Decide whether the ANSWER satisfies the RUBRIC. Do not reward answers that "
    "are merely plausible-sounding; grade only against the rubric. "
    "First write ONE short line of reasoning. Then, on its own final line, "
    "output exactly `VERDICT: PASS` or `VERDICT: FAIL`."
)

_VERDICT_RE = re.compile(r"VERDICT:\s*(PASS|FAIL)", re.IGNORECASE)


def _build_prompt(question: str, response: str, rubric: str) -> str:
    return (
        f"QUESTION:\n{question}\n\n"
        f"ANSWER:\n{response}\n\n"
        f"RUBRIC:\n{rubric}\n\n"
        "Grade the answer now."
    )


async def llm_judge(judge: Provider, question: str, response: str, rubric: str) -> GradeResult:
    if not response.strip():
        return GradeResult(False, "empty answer")
    completion = await judge.complete(
        _build_prompt(question, response, rubric), system=JUDGE_SYSTEM
    )
    match = _VERDICT_RE.search(completion.text)
    if not match:
        snippet = completion.text.strip()[:120]
        return GradeResult(False, f"judge returned no VERDICT line: {snippet!r}")
    passed = match.group(1).upper() == "PASS"
    reason = completion.text[: match.start()].strip().replace("\n", " ")
    return GradeResult(passed, reason[:200])

"""Graders decide whether a model's answer passes a test case.

Each grader is `(response_text, expected) -> GradeResult`. Cheap deterministic
graders live here; LLM-as-judge will be a separate async grader (planned).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict


@dataclass
class GradeResult:
    passed: bool
    detail: str = ""


def exact_match(response: str, expected: Any) -> GradeResult:
    got = response.strip().casefold()
    want = str(expected).strip().casefold()
    return GradeResult(got == want, f"expected {want!r}, got {got!r}" if got != want else "")


def contains(response: str, expected: Any) -> GradeResult:
    ok = str(expected).casefold() in response.casefold()
    return GradeResult(ok, "" if ok else f"{expected!r} not found in response")


def regex(response: str, expected: Any) -> GradeResult:
    ok = re.search(str(expected), response) is not None
    return GradeResult(ok, "" if ok else f"pattern {expected!r} did not match")


def json_fields(response: str, expected: Any) -> GradeResult:
    """Parse the first JSON object in the response; every expected key must match."""
    match = re.search(r"\{.*\}", response, re.DOTALL)
    if not match:
        return GradeResult(False, "no JSON object in response")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        return GradeResult(False, f"invalid JSON: {e}")
    if not isinstance(expected, dict):
        return GradeResult(False, "expected value for json_fields grader must be a mapping")
    mismatches = [
        f"{key}: expected {want!r}, got {data.get(key)!r}"
        for key, want in expected.items()
        if data.get(key) != want
    ]
    return GradeResult(not mismatches, "; ".join(mismatches))


GRADERS: Dict[str, Callable[[str, Any], GradeResult]] = {
    "exact_match": exact_match,
    "contains": contains,
    "regex": regex,
    "json_fields": json_fields,
}

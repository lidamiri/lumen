from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from lumen.errors import InputError


@dataclass(frozen=True)
class EvalCase:
    id: str
    question: str
    expected: tuple[str, ...] = ()
    reference: str = ""
    document_ids: tuple[str, ...] = ()


def load_dataset(path: str | Path) -> list[EvalCase]:
    source = Path(path)
    cases: list[EvalCase] = []
    for number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise InputError(f"{source}:{number}: invalid JSON ({exc.msg})") from exc
        question = raw.get("question")
        if not isinstance(question, str) or not question.strip():
            raise InputError(f"{source}:{number}: 'question' must be a non-empty string")
        cases.append(
            EvalCase(
                id=str(raw.get("id", number)),
                question=question.strip(),
                expected=tuple(str(item) for item in raw.get("expected", [])),
                reference=str(raw.get("reference", "")),
                document_ids=tuple(str(item) for item in raw.get("document_ids", [])),
            )
        )
    if not cases:
        raise InputError(f"{source} contains no evaluation cases")
    return cases

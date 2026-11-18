from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from lumen.eval.dataset import EvalCase
from lumen.eval.metrics import keyword_recall, token_f1
from lumen.service import LumenService


@dataclass(frozen=True)
class CaseResult:
    id: str
    question: str
    answer: str
    keyword_recall: float
    token_f1: float | None
    grounded: bool
    sources: int
    latency_ms: float


@dataclass(frozen=True)
class EvalReport:
    results: tuple[CaseResult, ...]

    def summary(self) -> dict[str, Any]:
        f1_values = [result.token_f1 for result in self.results if result.token_f1 is not None]
        return {
            "cases": len(self.results),
            "mean_keyword_recall": _mean([result.keyword_recall for result in self.results]),
            "mean_token_f1": _mean(f1_values) if f1_values else None,
            "grounded_rate": _mean([1.0 if result.grounded else 0.0 for result in self.results]),
            "mean_latency_ms": _mean([result.latency_ms for result in self.results]),
        }

    def to_dict(self) -> dict[str, Any]:
        return {"summary": self.summary(), "results": [asdict(result) for result in self.results]}


class EvalRunner:
    def __init__(self, service: LumenService) -> None:
        self._service = service

    def run(self, cases: Sequence[EvalCase]) -> EvalReport:
        results = []
        for case in cases:
            started = time.perf_counter()
            answer = self._service.ask(
                case.question,
                document_ids=frozenset(case.document_ids) or None,
            )
            elapsed_ms = (time.perf_counter() - started) * 1000
            results.append(
                CaseResult(
                    id=case.id,
                    question=case.question,
                    answer=answer.answer,
                    keyword_recall=round(keyword_recall(answer.answer, case.expected), 4),
                    token_f1=round(token_f1(answer.answer, case.reference), 4) if case.reference else None,
                    grounded=answer.grounded,
                    sources=len(answer.sources),
                    latency_ms=round(elapsed_ms, 2),
                )
            )
        return EvalReport(results=tuple(results))


def _mean(values: Sequence[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0

from __future__ import annotations

import re
from collections import Counter

_TOKEN = re.compile(r"\w+")


def normalize(text: str) -> str:
    return " ".join(_TOKEN.findall(text.lower()))


def keyword_recall(answer: str, expected: tuple[str, ...]) -> float:
    if not expected:
        return 1.0
    haystack = f" {normalize(answer)} "
    hits = 0
    for item in expected:
        needle = normalize(item)
        if needle and f" {needle} " in haystack:
            hits += 1
    return hits / len(expected)


def token_f1(prediction: str, reference: str) -> float:
    predicted = _TOKEN.findall(prediction.lower())
    gold = _TOKEN.findall(reference.lower())
    if not predicted or not gold:
        return float(predicted == gold)
    overlap = sum((Counter(predicted) & Counter(gold)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(gold)
    return 2 * precision * recall / (precision + recall)

import json

import pytest

from lumen.errors import InputError
from lumen.eval.dataset import EvalCase, load_dataset
from lumen.eval.metrics import keyword_recall, token_f1
from lumen.eval.runner import EvalRunner
from lumen.models import Document


def test_keyword_recall_matches_whole_words_only():
    assert keyword_recall("The warranty is 24 months", ("24 months", "warranty")) == 1.0
    assert keyword_recall("nothing relevant", ("24",)) == 0.0
    assert keyword_recall("124 items", ("24",)) == 0.0
    assert keyword_recall("anything", ()) == 1.0


def test_token_f1_scores_overlap():
    assert token_f1("the cat sat", "the cat sat") == 1.0
    assert token_f1("a b", "c d") == 0.0
    assert token_f1("a b", "a c") == pytest.approx(0.5)


def test_load_dataset_parses_jsonl(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"id": "q1", "question": "How long?", "expected": ["24"]}),
                "",
                json.dumps({"question": "Shipping?", "reference": "free shipping"}),
            ]
        ),
        encoding="utf-8",
    )
    cases = load_dataset(path)
    assert [case.id for case in cases] == ["q1", "3"]
    assert cases[0].expected == ("24",)
    assert cases[1].reference == "free shipping"


def test_load_dataset_rejects_invalid_lines(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(InputError):
        load_dataset(path)


def test_load_dataset_requires_cases(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("\n", encoding="utf-8")
    with pytest.raises(InputError):
        load_dataset(path)


def test_runner_reports_grounding_and_recall(make_service):
    service = make_service()
    service.ingest([Document(id="guide.md", text="The warranty lasts 24 months.")])
    report = EvalRunner(service).run(
        [
            EvalCase(id="w", question="How long is the warranty?", expected=("warranty",)),
            EvalCase(id="x", question="Unrelated?", expected=("refund",)),
        ]
    )
    summary = report.summary()
    assert summary["cases"] == 2
    assert summary["grounded_rate"] == 1.0
    assert report.results[0].keyword_recall == 1.0
    assert report.results[1].keyword_recall == 0.0
    assert report.to_dict()["results"][0]["id"] == "w"

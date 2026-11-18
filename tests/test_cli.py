import json

import pytest

from lumen.cli import build_parser, main
from lumen.config import get_settings


@pytest.fixture
def cli_env(tmp_path, monkeypatch):
    monkeypatch.setenv("LUMEN_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("LUMEN_CACHE_ENABLED", "false")
    monkeypatch.setenv("LUMEN_EMBEDDING_DIMENSIONS", "128")
    monkeypatch.setenv("LUMEN_CHUNK_SIZE", "200")
    monkeypatch.setenv("LUMEN_CHUNK_OVERLAP", "20")
    get_settings.cache_clear()
    yield tmp_path
    get_settings.cache_clear()


def test_parser_requires_a_command():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_ask_accepts_repeated_document_filters():
    args = build_parser().parse_args(["ask", "q", "--document", "a.md", "--document", "b.md"])
    assert args.documents == ["a.md", "b.md"]


def test_ingest_ask_and_status_commands(cli_env, capsys):
    docs = cli_env / "docs"
    docs.mkdir()
    (docs / "guide.md").write_text("The warranty lasts 24 months.", encoding="utf-8")

    assert main(["ingest", str(docs)]) == 0
    assert json.loads(capsys.readouterr().out)["chunks"] == 1

    assert main(["ask", "How long is the warranty?"]) == 0
    output = capsys.readouterr().out
    assert "guide.md" in output

    assert main(["status"]) == 0
    assert json.loads(capsys.readouterr().out)["provider"] == "echo"


def test_missing_path_returns_error_code(cli_env, capsys):
    assert main(["ingest", str(cli_env / "nope")]) == 2
    assert "does not exist" in capsys.readouterr().err


def test_eval_command_writes_report(cli_env, capsys):
    docs = cli_env / "docs"
    docs.mkdir()
    (docs / "faq.md").write_text("Shipping is free above fifty dollars.", encoding="utf-8")
    assert main(["ingest", str(docs)]) == 0
    dataset = cli_env / "cases.jsonl"
    dataset.write_text(json.dumps({"id": "s", "question": "Is shipping free?", "expected": ["shipping"]}) + "\n", encoding="utf-8")
    report = cli_env / "reports" / "eval.json"
    assert main(["eval", str(dataset), "--report", str(report)]) == 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["summary"]["cases"] == 1
    assert payload["results"][0]["id"] == "s"

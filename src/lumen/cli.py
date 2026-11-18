from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from lumen import __version__
from lumen.config import get_settings
from lumen.container import build_container
from lumen.errors import LumenError
from lumen.eval.dataset import load_dataset
from lumen.eval.runner import EvalRunner
from lumen.logging_utils import configure_logging
from lumen.models import Message, Role
from lumen.rag.loaders import load_documents
from lumen.service import LumenService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lumen", description="Lumen LLM toolkit")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)

    ingest = commands.add_parser("ingest", help="index .txt, .md and .rst files into the vector store")
    ingest.add_argument("path", type=Path)

    ask = commands.add_parser("ask", help="answer a question grounded in indexed documents")
    ask.add_argument("question")
    ask.add_argument("--top-k", type=int, default=None)
    ask.add_argument("--document", action="append", dest="documents", default=None)

    complete = commands.add_parser("complete", help="send a single prompt to the configured provider")
    complete.add_argument("prompt")
    complete.add_argument("--system", default=None)

    agent = commands.add_parser("agent", help="answer a question with the tool-using agent")
    agent.add_argument("question")
    agent.add_argument("--max-steps", type=int, default=None)

    summarize = commands.add_parser("summarize", help="summarize a text file")
    summarize.add_argument("path", type=Path)
    summarize.add_argument("--max-sentences", type=int, default=3)

    classify = commands.add_parser("classify", help="label a text file with one of the given labels")
    classify.add_argument("path", type=Path)
    classify.add_argument("--labels", required=True, help="comma separated labels")

    delete = commands.add_parser("delete", help="remove a document from the vector store")
    delete.add_argument("document_id")

    evaluate = commands.add_parser("eval", help="run an evaluation dataset")
    evaluate.add_argument("dataset", type=Path)
    evaluate.add_argument("--report", type=Path, default=None)

    serve = commands.add_parser("serve", help="start the HTTP API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    commands.add_parser("status", help="show provider, store and tool status")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(get_settings().log_level)
    if args.command == "serve":
        return _serve(args.host, args.port)
    container = build_container()
    try:
        return _run(LumenService(container), args)
    except (LumenError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    finally:
        container.close()


def _run(service: LumenService, args: argparse.Namespace) -> int:
    settings = service.settings
    if args.command == "ingest":
        documents = load_documents(args.path, max_chars=settings.max_input_chars)
        if not documents:
            print("no supported documents found", file=sys.stderr)
            return 1
        _print_json(asdict(service.ingest(documents)))
        return 0
    if args.command == "ask":
        document_ids = frozenset(args.documents) if args.documents else None
        answer = service.ask(args.question, top_k=args.top_k, document_ids=document_ids)
        print(answer.answer)
        for number, hit in enumerate(answer.sources, start=1):
            print(f"[{number}] {hit.chunk.document_id} (score {hit.score:.3f})")
        return 0
    if args.command == "complete":
        messages: list[Message] = []
        if args.system:
            messages.append(Message(role=Role.SYSTEM, content=args.system))
        messages.append(Message(role=Role.USER, content=args.prompt))
        print(service.chat(messages).text)
        return 0
    if args.command == "agent":
        result = service.run_agent(args.question, max_steps=args.max_steps)
        for step in result.steps:
            print(f"tool {step.tool}({json.dumps(step.arguments, ensure_ascii=False)}) -> {step.output}")
        print(result.answer or f"(stopped: {result.stop_reason})")
        return 0
    if args.command == "summarize":
        text = args.path.read_text(encoding="utf-8")
        print(service.summarize(text, max_sentences=args.max_sentences))
        return 0
    if args.command == "classify":
        labels = [label for label in args.labels.split(",") if label.strip()]
        print(service.classify(args.path.read_text(encoding="utf-8"), labels))
        return 0
    if args.command == "delete":
        removed = service.delete_document(args.document_id)
        _print_json({"document_id": args.document_id, "removed_chunks": removed})
        return 0 if removed else 1
    if args.command == "eval":
        report = EvalRunner(service).run(load_dataset(args.dataset))
        payload = report.to_dict()
        _print_json(payload["summary"])
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return 0
    if args.command == "status":
        _print_json(service.health())
        return 0
    raise SystemExit(f"unknown command: {args.command}")


def _serve(host: str, port: int) -> int:
    import uvicorn

    from lumen.api.app import create_app

    uvicorn.run(create_app(), host=host, port=port)
    return 0


def _print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))

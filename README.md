# Lumen

Lumen is a Python toolkit for building applications on top of large language models. It provides a provider-neutral chat layer for OpenAI and Anthropic, a retrieval-augmented question answering pipeline with a built-in vector store, a tool-using agent loop, prompt templates, an evaluation harness, a command-line interface and a versioned HTTP API.

The project runs fully offline by default. The built-in echo provider and the deterministic hashing embedder let every part of the system be exercised without network access or API keys. Switching to a hosted model is a configuration change.

## Features

- Provider adapters for the OpenAI chat completions API and the Anthropic messages API, including native tool calling
- Retrying HTTP client with exponential backoff and jitter for transient failures
- Token-bucket rate limiting and a SQLite completion cache applied as transparent wrappers
- Prompt templates with validation of missing placeholders and a registry of built-in prompts
- Document ingestion for text, Markdown and reStructuredText files with overlapping sentence-aware chunking
- Embeddings through a deterministic hashing embedder or the OpenAI embeddings endpoint
- Vector stores: in-memory and SQLite, both with exact cosine search and per-document filtering
- Grounded question answering that returns numbered sources and refuses to answer when no relevant passage exists
- Agent loop with calculator, clock and document search tools, bounded by step count and token budget
- Evaluation harness with JSONL datasets, keyword recall and token F1 metrics, and JSON reports
- Command-line interface and a FastAPI application with API key authentication
- Dockerfile, Makefile and a CI workflow that runs the test suite on Python 3.10 to 3.12

## Requirements

- Python 3.10 or newer
- Git, if you want to work with the commit history
- An OpenAI or Anthropic API key only when using those providers

## Installation

Create a virtual environment, install the package with its development dependencies, and run the test suite:

```bash
make install
make test
```

The same steps without make:

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
```

## Configuration

Settings are read from environment variables prefixed with `LUMEN_` and from a `.env` file in the working directory. Copy the template and edit it as needed:

```bash
cp .env.example .env
```

The most important settings are listed below. Every setting has a default, so none of them are required for the offline demo.

| Variable | Default | Purpose |
| --- | --- | --- |
| `LUMEN_PROVIDER` | `echo` | Chat backend: `echo`, `openai` or `anthropic` |
| `LUMEN_MODEL` | `gpt-4o-mini` | Model name sent to the provider |
| `OPENAI_API_KEY` | unset | Required for the `openai` provider and the OpenAI embedder |
| `ANTHROPIC_API_KEY` | unset | Required for the `anthropic` provider |
| `LUMEN_EMBEDDING_PROVIDER` | `hashing` | `hashing` (offline) or `openai` |
| `LUMEN_EMBEDDING_DIMENSIONS` | `384` | Vector size; a store cannot be reopened with a different size |
| `LUMEN_DATA_DIR` | `data` | Directory for the vector database and the completion cache |
| `LUMEN_CHUNK_SIZE` | `800` | Maximum characters per chunk |
| `LUMEN_CHUNK_OVERLAP` | `120` | Characters carried from one chunk into the next |
| `LUMEN_TOP_K` | `4` | Passages retrieved per question |
| `LUMEN_CACHE_ENABLED` | `true` | Serve identical completion requests from the local cache |
| `LUMEN_RATE_LIMIT_PER_MINUTE` | `120` | Maximum provider calls per minute |
| `LUMEN_AGENT_MAX_STEPS` | `6` | Maximum model turns per agent run |
| `LUMEN_AGENT_TOKEN_BUDGET` | `20000` | Token ceiling per agent run |
| `LUMEN_MAX_INPUT_CHARS` | `20000` | Largest accepted text per request |
| `LUMEN_API_KEY` | unset | When set, the API requires the `X-API-Key` header |
| `LUMEN_LOG_LEVEL` | `INFO` | Python logging level |

## Command-line usage

After installation the `lumen` command is available inside the virtual environment. The examples below use the bundled documents in `examples/docs`.

Index a directory of documents:

```bash
.venv/bin/lumen ingest examples/docs
```

Ask a question grounded in the indexed documents. Use `--document` to restrict the search to specific document identifiers:

```bash
.venv/bin/lumen ask "How long is the device warranty?"
.venv/bin/lumen ask "How fast does priority support respond?" --document warranty-and-support.md
```

Send a single prompt to the configured model:

```bash
.venv/bin/lumen complete "Explain vector similarity in two sentences." --system "Answer briefly."
```

Run the tool-using agent:

```bash
.venv/bin/lumen agent "What is 12 * (3 + 4), and what time is it in Tokyo?"
```

Summarise and classify text files:

```bash
.venv/bin/lumen summarize examples/docs/returns-policy.md --max-sentences 2
.venv/bin/lumen classify examples/docs/returns-policy.md --labels "policy,marketing,technical"
```

Remove a document from the index and inspect the system:

```bash
.venv/bin/lumen delete returns-policy.md
.venv/bin/lumen status
```

Run the evaluation dataset and write a report:

```bash
.venv/bin/lumen eval examples/eval.jsonl --report reports/eval.json
```

With the default echo provider the answers simply repeat the prompt, so the evaluation measures retrieval quality only. Configure a real provider to measure answer quality.

## HTTP API

Start the server:

```bash
.venv/bin/lumen serve --host 127.0.0.1 --port 8000
```

Interactive documentation is generated at `http://127.0.0.1:8000/docs`. The main endpoints are:

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Status, provider, chunk count and available tools (no authentication) |
| `POST` | `/v1/chat` | Chat completion over a list of messages |
| `POST` | `/v1/documents` | Ingest documents supplied in the request body |
| `DELETE` | `/v1/documents/{document_id}` | Remove a document and its chunks |
| `POST` | `/v1/ask` | Grounded question answering with sources |
| `POST` | `/v1/agent` | Run the tool-using agent |
| `POST` | `/v1/summaries` | Summarise a text |
| `POST` | `/v1/classifications` | Choose one label from a list |

Example requests:

```bash
curl -s http://127.0.0.1:8000/v1/documents \
  -H 'Content-Type: application/json' \
  -d '{"documents": [{"id": "faq.md", "text": "Shipping is free above fifty dollars."}]}'

curl -s http://127.0.0.1:8000/v1/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "Is shipping free?"}'
```

When `LUMEN_API_KEY` is set, add `-H 'X-API-Key: <key>'` to each request except the health check.

## Docker

Build and run the image:

```bash
docker build -t lumen .
docker run --rm -p 8000:8000 -e OPENAI_API_KEY -e LUMEN_PROVIDER=openai -v lumen-data:/data lumen
```

## Project layout

```
src/lumen/
  api/              HTTP application, request and response schemas, authentication
  agent/            tool-calling agent loop
  embeddings/       embedding interface, hashing and OpenAI implementations
  eval/             dataset loader, metrics and evaluation runner
  prompts/          prompt templates, registry and built-in prompt library
  providers/        chat provider interface, adapters, HTTP helper and wrappers
  rag/              chunking, loaders, ingestion, retrieval and answer pipeline
  store/            vector store interface, in-memory and SQLite implementations
  tools/            tool interface, registry, calculator, clock and document search
  cache.py          SQLite completion cache
  cli.py            command-line entry point
  config.py         settings loaded from the environment
  container.py      construction of concrete components from settings
  errors.py         exception hierarchy
  models.py         immutable domain models
  ratelimit.py      token bucket rate limiter
  service.py        application facade used by the CLI and the API
tests/              unit and integration tests
examples/           sample documents and an evaluation dataset
docs/               architecture notes
```

See `docs/ARCHITECTURE.md` for a description of each layer and the extension points.

## Development

Run the linter-equivalent compile check and the tests:

```bash
make lint
make test
```

Adding a provider means implementing `ChatProvider` and registering it in `build_provider`. Adding a vector database means implementing `VectorStore`. Both are covered by the shared test helpers in `tests/helpers.py`.

## License

Released under the MIT License.

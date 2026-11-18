# Architecture

Lumen is organised as a set of layers. Each layer depends only on the layers below it, and every boundary that talks to the outside world is an abstract interface with at least one concrete implementation.

## Layers

### Domain models

`lumen.models` defines the immutable data types shared by every layer: messages, tool calls, completions, usage counters, documents, chunks and search hits. Nothing in this module performs I/O.

### Configuration

`lumen.config.Settings` is a pydantic settings model. Every field can be set with an environment variable prefixed with `LUMEN_`, or through a `.env` file. Provider keys also accept the conventional `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` names. Validation runs once at start-up, so invalid combinations such as a chunk overlap larger than the chunk size fail fast.

### Providers

`lumen.providers.base.ChatProvider` is the single interface every language model backend implements. Two network adapters ship with the project:

- `OpenAIProvider` speaks the chat completions protocol, including native function calling.
- `AnthropicProvider` speaks the messages protocol, converting system prompts, tool use blocks and tool results.

`EchoProvider` is a deterministic offline backend used for development, tests and demonstrations.

Two wrappers add cross-cutting behaviour without changing the interface:

- `RateLimitedProvider` throttles calls with a token bucket from `lumen.ratelimit`.
- `CachingProvider` stores completions in SQLite, keyed by a SHA-256 digest of the provider, model, messages, sampling parameters and tool definitions.

`lumen.providers.http.post_json` performs the HTTP call with exponential backoff and jitter. It retries timeouts, connection errors and the status codes in `RETRYABLE_STATUS`, and fails immediately on other client errors.

### Prompts

`lumen.prompts.template.PromptTemplate` renders system and user messages from named placeholders. Missing placeholders are reported before any call is made, and substituted values are never re-parsed as templates. `PromptRegistry` holds named templates, and `lumen.prompts.library` provides the built-in ones.

### Embeddings

`lumen.embeddings.base.Embedder` produces unit-length float32 matrices.

- `HashingEmbedder` uses signed feature hashing over unigrams and bigrams. It needs no model download and is fully deterministic.
- `OpenAIEmbedder` calls the embeddings endpoint in batches and restores the response order by index.

### Vector stores

`lumen.store.base.VectorStore` defines upsert, similarity search with an optional document filter, document deletion and counting. Both implementations share the same ranking function in `lumen.store.base.rank`.

- `InMemoryVectorStore` keeps a dense matrix in memory and rebuilds it lazily after writes.
- `SQLiteVectorStore` persists chunks and vectors in a single file. It records the embedding dimensionality in a metadata table and refuses to open a store whose dimensionality differs from the configured one.

Search is exact (brute force). For corpora in the low hundreds of thousands of chunks this is fast enough; a larger deployment would replace the store behind the same interface.

### Retrieval-augmented generation

`lumen.rag` contains the pipeline in four steps:

1. `loaders` reads `.txt`, `.md` and `.rst` files and assigns each document a stable identifier relative to the ingestion root.
2. `chunking` splits text on sentence and paragraph boundaries, hard-wraps over-long words, and carries an overlap into the next chunk.
3. `ingest.Ingestor` replaces any previous chunks of a document, embeds the new chunks in batches and writes them to the store.
4. `pipeline.RagPipeline` retrieves hits, packs them into a numbered context block up to a character budget, renders the `rag_answer` template and returns the answer together with the sources used. When no hit is found the model is not called and the answer states that the system does not know.

### Tools and agents

`lumen.tools` provides a `Tool` base class and a `ToolRegistry`. Built-in tools are:

- `calculator`, which parses arithmetic with the `ast` module and evaluates only whitelisted nodes. It never calls `eval`.
- `current_time`, which reports the time in any IANA timezone.
- `search_documents`, which exposes the retriever to the model.

Tool failures are converted into text observations, so the model can recover instead of the request failing.

`lumen.agent.loop.Agent` runs the native tool-calling protocol. Each iteration calls the provider, executes the requested tools, appends the results and continues. The loop stops when the model answers without tool calls, when the step limit is reached, or when the token budget is exhausted. The result always records the stop reason and every step taken.

### Evaluation

`lumen.eval` loads JSONL datasets, scores answers with keyword recall (whole-word containment) and token F1 against an optional reference, and writes a JSON report with per-case details and aggregate figures.

### Service and composition

`lumen.container.build_container` wires settings into concrete objects. `lumen.service.LumenService` is the facade used by both the CLI and the HTTP API. It enforces the input size limit and exposes chat, ingestion, question answering, agent runs, summarisation, classification and health reporting.

### Interfaces

- `lumen.cli` provides the `lumen` command with subcommands for ingestion, questions, completions, agents, summaries, classification, deletion, evaluation, serving and status.
- `lumen.api.app.create_app` builds a FastAPI application. Routes live under `/v1`, the health check is public, and an API key is required when `LUMEN_API_KEY` is set. Domain errors map to HTTP status codes: invalid input to 422, unknown resources to 404, upstream provider failures to 502, and configuration problems to 503.

## Request flow for a grounded question

1. The API or CLI calls `LumenService.ask`.
2. The service checks the input size and delegates to `RagPipeline.answer`.
3. The retriever embeds the question and asks the store for the top matching chunks.
4. The pipeline builds the context block and renders the prompt.
5. The provider stack is called: caching first, then rate limiting, then the network adapter.
6. The answer and its sources are returned to the caller.

## Extension points

- A new language model backend implements `ChatProvider` and is registered in `lumen.container.build_provider`.
- A new embedding backend implements `Embedder` and is registered in `lumen.container.build_embedder`.
- A new vector database implements `VectorStore`.
- A new tool subclasses `Tool` and is added to `lumen.tools.builtin.default_tools`.
- A new prompt is added to `lumen.prompts.library.BUILTIN_TEMPLATES`.

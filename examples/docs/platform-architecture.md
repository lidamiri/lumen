Platform Architecture

The ingestion service splits documents into overlapping chunks of about 800 characters. Each chunk is embedded and stored in the vector database together with its document identifier and metadata.

Retrieval embeds the user question with the same embedder and returns the top matching chunks ranked by cosine similarity.

Requests to the language model are rate limited with a token bucket. Identical requests are served from a local completion cache when caching is enabled.

The agent loop lets the model call tools such as a calculator, a clock and document search. Each run is bounded by a maximum number of steps and a token budget.

Every HTTP request to the versioned API is authenticated with an API key when one is configured. Health checks are always public.

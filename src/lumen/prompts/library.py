from lumen.prompts.template import PromptRegistry, PromptTemplate

BUILTIN_TEMPLATES: tuple[PromptTemplate, ...] = (
    PromptTemplate(
        name="rag_answer",
        system=(
            "You answer questions using only the numbered context passages provided. "
            "Cite the passages you rely on with their bracketed numbers, for example [2]. "
            "If the context does not contain the answer, say that you do not know. "
            "Never use outside knowledge."
        ),
        user="Context:\n{context}\n\nQuestion: {question}\n\nAnswer:",
    ),
    PromptTemplate(
        name="summarize",
        system="You write precise, faithful summaries and never add facts that are not in the text.",
        user="Summarize the following text in at most {max_sentences} sentences.\n\n{text}",
    ),
    PromptTemplate(
        name="classify",
        system=(
            "You label text with exactly one of the given categories. "
            "Reply with the category label only."
        ),
        user="Categories: {labels}\n\nText: {text}\n\nLabel:",
    ),
    PromptTemplate(
        name="rewrite_query",
        system=(
            "Rewrite the user question as a standalone search query that keeps every "
            "important term. Reply with the query only."
        ),
        user="{question}",
    ),
)


def default_registry() -> PromptRegistry:
    registry = PromptRegistry()
    for template in BUILTIN_TEMPLATES:
        registry.register(template)
    return registry

import pytest

from lumen.errors import NotFoundError
from lumen.models import Role
from lumen.prompts.library import default_registry
from lumen.prompts.template import PromptRegistry, PromptTemplate, TemplateError


def test_render_substitutes_values():
    template = PromptTemplate(name="t", system="system {a}", user="user {b}")
    assert template.render(a="1", b="2") == ("system 1", "user 2")


def test_missing_variable_is_reported():
    template = PromptTemplate(name="t", system="", user="{a} {b}")
    with pytest.raises(TemplateError, match="b"):
        template.render(a="x")


def test_build_skips_blank_system_message():
    messages = PromptTemplate(name="t", system="  ", user="{q}").build(q="hello")
    assert [message.role for message in messages] == [Role.USER]


def test_values_are_not_reinterpreted_as_fields():
    template = PromptTemplate(name="t", system="", user="{q}")
    _, user = template.render(q="{not_a_field}")
    assert user == "{not_a_field}"


def test_duplicate_registration_is_rejected():
    registry = PromptRegistry()
    template = PromptTemplate(name="t", system="", user="x")
    registry.register(template)
    with pytest.raises(TemplateError):
        registry.register(template)


def test_unknown_template_raises_not_found():
    with pytest.raises(NotFoundError):
        default_registry().get("missing")


def test_default_registry_contains_builtin_templates():
    registry = default_registry()
    for name in ("rag_answer", "summarize", "classify", "rewrite_query"):
        assert name in registry

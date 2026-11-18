from __future__ import annotations

import string
from dataclasses import dataclass

from lumen.errors import NotFoundError
from lumen.models import Message, Role


class TemplateError(ValueError):
    pass


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    system: str
    user: str

    def fields(self) -> frozenset[str]:
        found: set[str] = set()
        for text in (self.system, self.user):
            for _, field_name, _, _ in string.Formatter().parse(text):
                if field_name:
                    found.add(field_name.split(".")[0].split("[")[0])
        return frozenset(found)

    def render(self, **values: str) -> tuple[str, str]:
        missing = sorted(name for name in self.fields() if name not in values)
        if missing:
            raise TemplateError(f"template {self.name!r} is missing: {', '.join(missing)}")
        return self.system.format(**values), self.user.format(**values)

    def build(self, **values: str) -> list[Message]:
        system, user = self.render(**values)
        messages: list[Message] = []
        if system.strip():
            messages.append(Message(role=Role.SYSTEM, content=system))
        messages.append(Message(role=Role.USER, content=user))
        return messages


class PromptRegistry:
    def __init__(self) -> None:
        self._templates: dict[str, PromptTemplate] = {}

    def register(self, template: PromptTemplate) -> None:
        if template.name in self._templates:
            raise TemplateError(f"template {template.name!r} is already registered")
        self._templates[template.name] = template

    def get(self, name: str) -> PromptTemplate:
        try:
            return self._templates[name]
        except KeyError as exc:
            raise NotFoundError(f"unknown prompt template: {name}") from exc

    def names(self) -> list[str]:
        return sorted(self._templates)

    def __contains__(self, name: object) -> bool:
        return name in self._templates

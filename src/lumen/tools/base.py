from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from lumen.models import ToolSpec


class Tool(ABC):
    name: str
    description: str
    parameters: dict[str, Any]

    def spec(self) -> ToolSpec:
        return ToolSpec(name=self.name, description=self.description, parameters=self.parameters)

    @abstractmethod
    def run(self, arguments: dict[str, Any]) -> str:
        raise NotImplementedError

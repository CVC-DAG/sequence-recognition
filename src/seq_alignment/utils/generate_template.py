"""Given a BaseModel config definition, generate a JSON config template."""

from __future__ import annotations

from pydantic import BaseModel
from typing import Type


class TemplateGenerator:
    def __init__(self, config_object: Type) -> None:
        self._config_object = config_object

    @classmethod
    def _run_type_tree(node: Type) -> object:
        output = {}
        for key, tt in node.__annotations__:
            if tt is BaseModel:
                output[key] = TemplateGenerator._run_type_tree()
            else:
                output[key] = f"{tt.__name__}"
        return output

    def generate(self):
        return TemplateGenerator._run_type_tree(self._config_object)

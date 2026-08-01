from __future__ import annotations

from typing import Any


def make_strict_json_schema(value: Any) -> Any:
    """Recursively reject undeclared object properties in a JSON schema."""
    if isinstance(value, dict):
        strict = {key: make_strict_json_schema(item) for key, item in value.items()}
        if strict.get("type") == "object":
            strict["additionalProperties"] = False
        return strict
    if isinstance(value, list):
        return [make_strict_json_schema(item) for item in value]
    return value

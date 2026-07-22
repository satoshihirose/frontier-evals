from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from paperbench.requirements import (
    REQUIREMENTS_CONTAINER_PATH,
    REQUIREMENTS_INSTRUCTION,
    add_requirements_instruction,
    load_requirements_input,
)


def write_requirements_csv(path: Path) -> bytes:
    content = (
        "\ufeffrequirement_id,requirement_statement,procedure_category,resolution_state\n"
        "req-001,The evaluation must use the full test set.,Experimental Inputs,paper_resolved\n"
        "req-002,The parser must stop before the next Q: marker.,Method Configuration,"
        "neighbor_resolved\n"
    ).encode("utf-8")
    path.write_bytes(content)
    return content


def test_load_requirements_input_validates_and_describes_csv(tmp_path: Path) -> None:
    path = tmp_path / "requirements.csv"
    content = write_requirements_csv(path)

    requirements = load_requirements_input(path)

    assert requirements.source_path == path.resolve()
    assert requirements.content == content
    assert requirements.sha256 == hashlib.sha256(content).hexdigest()
    assert requirements.requirement_count == 2


def test_load_requirements_input_rejects_missing_required_columns(tmp_path: Path) -> None:
    path = tmp_path / "requirements.csv"
    path.write_text("requirement_id,requirement_statement\nreq-001,A must be B.\n")

    with pytest.raises(ValueError, match="procedure_category"):
        load_requirements_input(path)


def test_add_requirements_instruction_is_noop_for_baseline() -> None:
    base = "Reproduce the paper.\n"

    assert add_requirements_instruction(base, requirements_provided=False) == base


def test_add_requirements_instruction_appends_fixed_agent_resource_notice() -> None:
    base = "Reproduce the paper.\n"

    result = add_requirements_instruction(base, requirements_provided=True)

    assert result.startswith(base)
    assert REQUIREMENTS_INSTRUCTION in result
    assert REQUIREMENTS_CONTAINER_PATH in result

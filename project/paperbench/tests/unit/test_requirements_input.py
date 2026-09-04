from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from paperbench.requirements import (
    REQUIREMENTS_CONTAINER_PATH,
    REQUIREMENTS_INSTRUCTION,
    add_requirements_instruction,
    add_self_generated_requirements_instruction,
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
    assert "extracted and normalized from the" in result
    assert "Each `requirement_statement` describes a condition" in result
    assert "must satisfy" in result
    assert "operational summary" in result
    assert "additional list" not in result.lower()


def test_self_generated_requirements_instruction_includes_task_constraints() -> None:
    result = add_self_generated_requirements_instruction(
        "Reproduce the paper.\n",
        self_generated=True,
    )
    normalized_result = " ".join(result.split())

    assert (
        "all reproduction-defining constraints stated in the task instructions above"
        in normalized_result
    )
    assert "concrete benchmark task contracts" in normalized_result
    assert "explicitly delegates that missing choice" in normalized_result
    assert "mere citation or attribution" in normalized_result
    assert "pure scope exclusions as filters" in normalized_result
    assert "appendix-only experiments" in normalized_result
    assert (
        "implementation details for an in-scope main-body experiment"
        in normalized_result
    )
    assert "explicitly corrects or replaces" in normalized_result
    assert "later source position alone" in normalized_result

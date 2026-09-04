import json
from typing import Any

from paperbench.constants import WORKSPACE_BASE

RUBRIC_CONTAINER_PATH = f"{WORKSPACE_BASE}/paper/rubric.json"
JUDGE_ADDENDUM_CONTAINER_PATH = f"{WORKSPACE_BASE}/paper/judge.addendum.md"
RUBRIC_CRITERIA_CONTAINER_PATH = f"{WORKSPACE_BASE}/paper/rubric_criteria.json"

RUBRIC_VISIBLE_INSTRUCTION = """## Additional evaluation specification

The file `rubric.json`, and `judge.addendum.md` when that file is present, describe the evaluation criteria that the paper reproduction is expected to satisfy.

Use these files as an additional specification when planning and implementing the reproduction, and aim to satisfy as many applicable criteria as possible.

When `judge.addendum.md` is present, it may contain benchmark-specific clarifications or corrections to `rubric.json`. Where they explicitly conflict, follow `judge.addendum.md`. Otherwise, use the target paper as the primary description of the method and use the evaluation specification to determine the required reproduction scope, experimental conditions, and outputs.

All other submission and reproduction instructions remain unchanged.
"""

RUBRIC_CRITERIA_INSTRUCTION = f"""## Evaluation criteria

`{RUBRIC_CRITERIA_CONTAINER_PATH}` contains the leaf evaluation criteria, copied verbatim from the evaluation rubric and flattened into one JSON array.

Use these criteria when planning, implementing, and validating the reproduction. The file does not include rubric weights, hierarchy, identifiers, or grader-only explanations. The paper and the other supplied task materials remain the authoritative description of the work.
"""


def flatten_rubric_criteria(rubric: dict[str, Any]) -> list[str]:
    """Return leaf requirement text in source tree order without rewriting it."""
    criteria: list[str] = []

    def visit(node: dict[str, Any]) -> None:
        children = node.get("sub_tasks")
        if not isinstance(children, list):
            raise ValueError("Every rubric node must contain a sub_tasks list")
        if not children:
            requirement = node.get("requirements")
            if not isinstance(requirement, str) or not requirement:
                raise ValueError("Every rubric leaf must contain non-empty requirements text")
            criteria.append(requirement)
            return
        for child in children:
            if not isinstance(child, dict):
                raise ValueError("Rubric sub_tasks must contain objects")
            visit(child)

    visit(rubric)
    return criteria


def serialize_rubric_criteria(rubric_content: bytes) -> bytes:
    rubric = json.loads(rubric_content)
    if not isinstance(rubric, dict):
        raise ValueError("Rubric root must be an object")
    criteria = flatten_rubric_criteria(rubric)
    return (json.dumps(criteria, ensure_ascii=False, indent=2) + "\n").encode()


def add_evaluation_specification_instruction(instructions: str, *, mode: str) -> str:
    if mode == "rubric-visible":
        appended = RUBRIC_VISIBLE_INSTRUCTION
    elif mode == "rubric-criteria":
        appended = RUBRIC_CRITERIA_INSTRUCTION
    else:
        return instructions
    return f"{instructions.rstrip()}\n\n{appended}"

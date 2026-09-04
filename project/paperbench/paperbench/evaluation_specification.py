import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from paperbench.constants import WORKSPACE_BASE

RUBRIC_CONTAINER_PATH = f"{WORKSPACE_BASE}/paper/rubric.json"
JUDGE_ADDENDUM_CONTAINER_PATH = f"{WORKSPACE_BASE}/paper/judge.addendum.md"
RUBRIC_CRITERIA_CONTAINER_PATH = f"{WORKSPACE_BASE}/paper/rubric_criteria.json"
RUBRIC_CRITERIA_ARTIFACT_DIRNAME = "rubric-criteria"
RUBRIC_CRITERIA_MANIFEST_FILENAME = "manifest.json"

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


@dataclass(frozen=True)
class RubricCriteriaArtifact:
    content: bytes
    criteria: tuple[str, ...]
    sha256: str


def _regeneration_error(detail: str) -> ValueError:
    return ValueError(
        f"{detail}; regenerate all rubric criteria with "
        "`python -m paperbench.scripts.generate_rubric_criteria`"
    )


def load_rubric_criteria_artifact(
    *, paper_id: str, rubric_path: Path
) -> RubricCriteriaArtifact:
    """Load a checked-in criteria artifact only when it matches its source rubric."""
    artifact_dir = rubric_path.parent.parent.parent / RUBRIC_CRITERIA_ARTIFACT_DIRNAME
    criteria_path = artifact_dir / f"{paper_id}.json"
    manifest_path = artifact_dir / RUBRIC_CRITERIA_MANIFEST_FILENAME
    try:
        manifest = json.loads(manifest_path.read_text())
        entry = manifest["papers"][paper_id]
        content = criteria_path.read_bytes()
    except (FileNotFoundError, KeyError, json.JSONDecodeError, TypeError) as error:
        raise _regeneration_error(f"Missing or invalid criteria artifact for {paper_id}") from error

    rubric_sha256 = hashlib.sha256(rubric_path.read_bytes()).hexdigest()
    artifact_sha256 = hashlib.sha256(content).hexdigest()
    if entry.get("rubric_sha256") != rubric_sha256:
        raise _regeneration_error(f"Source rubric changed for {paper_id}")
    if entry.get("rubric_criteria_sha256") != artifact_sha256:
        raise _regeneration_error(f"Criteria artifact changed for {paper_id}")

    try:
        decoded = json.loads(content)
    except json.JSONDecodeError as error:
        raise _regeneration_error(f"Criteria artifact is not valid JSON for {paper_id}") from error
    if not isinstance(decoded, list) or not decoded or not all(
        isinstance(criterion, str) and criterion for criterion in decoded
    ):
        raise _regeneration_error(f"Criteria artifact has invalid entries for {paper_id}")
    if entry.get("criterion_count") != len(decoded):
        raise _regeneration_error(f"Criteria count does not match for {paper_id}")

    return RubricCriteriaArtifact(
        content=content,
        criteria=tuple(decoded),
        sha256=artifact_sha256,
    )


def add_evaluation_specification_instruction(instructions: str, *, mode: str) -> str:
    if mode == "rubric-visible":
        appended = RUBRIC_VISIBLE_INSTRUCTION
    elif mode == "rubric-criteria":
        appended = RUBRIC_CRITERIA_INSTRUCTION
    else:
        return instructions
    return f"{instructions.rstrip()}\n\n{appended}"

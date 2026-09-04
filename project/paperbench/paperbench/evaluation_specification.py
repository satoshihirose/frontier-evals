from paperbench.constants import WORKSPACE_BASE

RUBRIC_CONTAINER_PATH = f"{WORKSPACE_BASE}/paper/rubric.json"
JUDGE_ADDENDUM_CONTAINER_PATH = f"{WORKSPACE_BASE}/paper/judge.addendum.md"

RUBRIC_VISIBLE_INSTRUCTION = """## Additional evaluation specification

The file `rubric.json`, and `judge.addendum.md` when that file is present, describe the evaluation criteria that the paper reproduction is expected to satisfy.

Use these files as an additional specification when planning and implementing the reproduction, and aim to satisfy as many applicable criteria as possible.

When `judge.addendum.md` is present, it may contain benchmark-specific clarifications or corrections to `rubric.json`. Where they explicitly conflict, follow `judge.addendum.md`. Otherwise, use the target paper as the primary description of the method and use the evaluation specification to determine the required reproduction scope, experimental conditions, and outputs.

All other submission and reproduction instructions remain unchanged.
"""


def add_evaluation_specification_instruction(
    instructions: str, *, rubric_visible: bool
) -> str:
    if not rubric_visible:
        return instructions
    return f"{instructions.rstrip()}\n\n{RUBRIC_VISIBLE_INSTRUCTION}"

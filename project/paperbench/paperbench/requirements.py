from __future__ import annotations

import csv
import hashlib
import io
from dataclasses import dataclass
from pathlib import Path

from paperbench.constants import WORKSPACE_BASE

REQUIREMENTS_CONTAINER_DIR = f"{WORKSPACE_BASE}/requirements"
REQUIREMENTS_CONTAINER_PATH = f"{REQUIREMENTS_CONTAINER_DIR}/requirements.csv"
REQUIRED_COLUMNS = frozenset(
    {
        "requirement_id",
        "requirement_statement",
        "procedure_category",
    }
)
REQUIREMENTS_INSTRUCTION = f"""

ADDITIONAL REPRODUCTION REQUIREMENTS
---
An additional list of reproduction requirements is available at
`{REQUIREMENTS_CONTAINER_PATH}`.

Read every row before implementation. Treat each `requirement_statement` as a condition that
the reproduction should satisfy. The list supplements, but does not replace, the paper,
addendum, task instructions, or submission rules.
""".lstrip()


@dataclass(frozen=True)
class RequirementsInput:
    source_path: Path
    content: bytes
    sha256: str
    requirement_count: int


def load_requirements_input(path: str | Path) -> RequirementsInput:
    source_path = Path(path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Requirements CSV not found: {source_path}")

    content = source_path.read_bytes()
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Requirements CSV must be UTF-8: {source_path}") from exc

    reader = csv.DictReader(io.StringIO(text, newline=""))
    fieldnames = set(reader.fieldnames or [])
    missing_columns = sorted(REQUIRED_COLUMNS - fieldnames)
    if missing_columns:
        raise ValueError(
            f"Requirements CSV is missing required columns {missing_columns}: {source_path}"
        )

    requirement_count = 0
    for row_number, row in enumerate(reader, start=2):
        if not any((value or "").strip() for value in row.values()):
            continue
        for column in REQUIRED_COLUMNS:
            if not (row.get(column) or "").strip():
                raise ValueError(
                    f"Requirements CSV has an empty {column!r} at row {row_number}: "
                    f"{source_path}"
                )
        requirement_count += 1

    return RequirementsInput(
        source_path=source_path,
        content=content,
        sha256=hashlib.sha256(content).hexdigest(),
        requirement_count=requirement_count,
    )


def add_requirements_instruction(base_instructions: str, *, requirements_provided: bool) -> str:
    if not requirements_provided:
        return base_instructions
    separator = "" if base_instructions.endswith("\n") else "\n"
    return f"{base_instructions}{separator}\n{REQUIREMENTS_INSTRUCTION}"

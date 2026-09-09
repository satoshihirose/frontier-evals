from __future__ import annotations

import json
import math
import time
from typing import Any

import blobfile as bf

from nanoeval.solvers.computer_tasks.code_execution_interface import ComputerInterface
from paperbench.constants import SUBMISSION_DIR

DEFAULT_MIN_REMAINING_SECONDS = 60 * 60
QUICK_UNCHANGED_MAX_SECONDS = 5 * 60
MAX_CONSECUTIVE_QUICK_UNCHANGED = 2
COMPLETION_REVIEW_METADATA = "completion-review.json"
INITIAL_SUBMISSION_FILENAME = "initial-submission.tar.gz"
SUBMISSION_GIT_HEAD_COMMAND = f"git -C {SUBMISSION_DIR} rev-parse HEAD"


def remaining_budget_seconds(
    *, time_limit_seconds: int | float, start_time: int | float, now: int | float
) -> int:
    return max(0, math.floor(time_limit_seconds - (now - start_time)))


def build_completion_review_prompt(remaining_seconds: int) -> str:
    if remaining_seconds < 0:
        raise ValueError("remaining_seconds must be non-negative")
    remaining_minutes = remaining_seconds // 60
    return (
        f"You have {remaining_minutes} minutes remaining. Continue working on the "
        "current task and use the remaining time to improve and verify your submission. "
        "If you determine that the task is already complete, perform any final "
        "verification you consider appropriate and finish."
    )


async def get_submission_git_head(computer: ComputerInterface) -> str | None:
    """Return the submission HEAD, or None when it cannot be observed safely."""
    try:
        result = await computer.send_shell_command(
            SUBMISSION_GIT_HEAD_COMMAND,
            idempotent=True,
        )
    except Exception:
        return None
    if result.exit_code != 0:
        return None
    head = result.output.decode(errors="replace").strip()
    return head if head else None


def is_quick_unchanged_review(
    *,
    head_before: str | None,
    head_after: str | None,
    started_at: int | float,
    finished_at: int | float,
) -> bool:
    """Detect a review that finished quickly without committing a change."""
    duration_seconds = max(0.0, finished_at - started_at)
    return (
        head_before is not None
        and head_after is not None
        and head_before == head_after
        and duration_seconds <= QUICK_UNCHANGED_MAX_SECONDS
    )


def latest_submission(run_dir: str) -> str | None:
    candidates = [
        path
        for path in bf.glob(bf.join(run_dir, "submissions", "*", "submission.tar.gz"))
        if not path.endswith("_executed.tar.gz")
    ]
    return max(candidates) if candidates else None


def snapshot_initial_submission(run_dir: str) -> str | None:
    source = latest_submission(run_dir)
    if source is None:
        return None
    destination = bf.join(run_dir, INITIAL_SUBMISSION_FILENAME)
    if not bf.exists(destination):
        bf.copy(source, destination)
    return destination


def write_completion_review_metadata(run_dir: str, payload: dict[str, Any]) -> None:
    path = bf.join(run_dir, COMPLETION_REVIEW_METADATA)
    existing: dict[str, Any] = {}
    if bf.exists(path):
        try:
            loaded = json.loads(bf.read_bytes(path))
            if isinstance(loaded, dict):
                existing = loaded
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass
    document = {
        "schema_version": 1,
        **existing,
        **payload,
        "updated_at": time.time(),
    }
    bf.write_bytes(
        path,
        (json.dumps(document, indent=2, sort_keys=True) + "\n").encode(),
    )

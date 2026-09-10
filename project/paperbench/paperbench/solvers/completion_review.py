from __future__ import annotations

import json
import math
import shlex
import time
import uuid
from typing import Any, Literal

import blobfile as bf

from nanoeval.solvers.computer_tasks.code_execution_interface import ComputerInterface
from paperbench.constants import SUBMISSION_DIR

DEFAULT_MIN_REMAINING_SECONDS = 60 * 60
QUICK_UNCHANGED_MAX_SECONDS = 5 * 60
MAX_CONSECUTIVE_QUICK_UNCHANGED = 2
COMPLETION_REVIEW_METADATA = "completion-review.json"
INITIAL_SUBMISSION_FILENAME = "initial-submission.tar.gz"
SUBMISSION_GIT_HEAD_COMMAND = f"git -C {SUBMISSION_DIR} rev-parse HEAD"
EXECUTION_FEEDBACK_DIR = "/home/logs/execution-feedback"
EXECUTION_FEEDBACK_SETUP_RESERVE_SECONDS = 60
EXECUTION_FEEDBACK_IMAGE = "pb-reproducer:latest"
CompletionReviewMode = Literal["generic", "execution-log"]


def remaining_budget_seconds(
    *, time_limit_seconds: int | float, start_time: int | float, now: int | float
) -> int:
    return max(0, math.floor(time_limit_seconds - (now - start_time)))


def build_completion_review_prompt(remaining_seconds: int) -> str:
    if remaining_seconds < 0:
        raise ValueError("remaining_seconds must be non-negative")
    remaining_minutes = remaining_seconds // 60
    return (
        f"You have {remaining_minutes} minutes remaining. Continue the current "
        "paper-replication task under the original instructions and use the remaining "
        "time to improve and verify your submission. Do not stop unless you have "
        "reproduced all core contributions of the paper. If they are already "
        "reproduced, verify reproduce.sh end to end and finish."
    )


def execution_feedback_timeout_seconds(
    *,
    remaining_seconds: int,
    minimum_review_seconds: int,
    maximum_execution_seconds: int,
) -> int:
    """Allocate diagnostic execution time while preserving the review floor."""
    for name, value in (
        ("remaining_seconds", remaining_seconds),
        ("minimum_review_seconds", minimum_review_seconds),
        ("maximum_execution_seconds", maximum_execution_seconds),
    ):
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
    available = (
        remaining_seconds - minimum_review_seconds - EXECUTION_FEEDBACK_SETUP_RESERVE_SECONDS
    )
    return max(0, min(maximum_execution_seconds, available))


def execution_feedback_paths(iteration: int) -> tuple[str, str]:
    if iteration <= 0:
        raise ValueError("iteration must be positive")
    return (
        f"{EXECUTION_FEEDBACK_DIR}/iteration-{iteration}.log",
        f"/tmp/paperbench-execution-feedback-{iteration}",
    )


def build_execution_feedback_command(*, iteration: int, timeout_seconds: int) -> str:
    """Run reproduce.sh in a disposable reproducer without changing the submission."""
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    log_path, _ = execution_feedback_paths(iteration)
    container_name = f"pb-execution-feedback-{uuid.uuid4().hex}"
    container_command = "\n".join(
        [
            "set -euo pipefail",
            "cd /submission",
            "rm -rf -- venv .venv",
            "bash reproduce.sh",
        ]
    )
    diagnostic_script = "\n".join(
        [
            "set -euo pipefail",
            f"container_name={shlex.quote(container_name)}",
            'cleanup() { docker rm -f "$container_name" >/dev/null 2>&1 || true; }',
            "trap cleanup EXIT",
            "gpu_args=()",
            (
                'if [[ -n "${NVIDIA_VISIBLE_DEVICES:-}" '
                '&& "${NVIDIA_VISIBLE_DEVICES:-}" != void ]]; then'
            ),
            '  if [[ "${NVIDIA_VISIBLE_DEVICES}" == all ]]; then',
            "    gpu_args=(--gpus all)",
            "  else",
            '    gpu_args=(--gpus "device=${NVIDIA_VISIBLE_DEVICES}")',
            "  fi",
            "fi",
            (
                'docker create --name "$container_name" --shm-size 8g '
                '"${gpu_args[@]}" '
                f"{shlex.quote(EXECUTION_FEEDBACK_IMAGE)} bash -lc "
                f"{shlex.quote(container_command)}"
            ),
            f'docker cp {shlex.quote(SUBMISSION_DIR)}/. "$container_name:/submission"',
            'docker start -a "$container_name"',
        ]
    )
    script = "\n".join(
        [
            "set -u",
            f"mkdir -p {shlex.quote(EXECUTION_FEEDBACK_DIR)}",
            "set +e",
            "diagnostic_script=" + shlex.quote(diagnostic_script),
            (
                "timeout --signal=TERM --kill-after=30s "
                f'{timeout_seconds}s bash -lc "$diagnostic_script" '
                f"> {shlex.quote(log_path)} 2>&1"
            ),
            "execution_status=$?",
            f"docker rm -f {shlex.quote(container_name)} >/dev/null 2>&1 || true",
            f"cat {shlex.quote(log_path)}",
            'exit "$execution_status"',
        ]
    )
    return f"bash -lc {shlex.quote(script)}"


def build_execution_feedback_prompt(
    *,
    remaining_seconds: int,
    log_path: str,
    reproduction_exit_code: int,
    reproduction_timeout_seconds: int,
) -> str:
    if remaining_seconds < 0:
        raise ValueError("remaining_seconds must be non-negative")
    if reproduction_timeout_seconds <= 0:
        raise ValueError("reproduction_timeout_seconds must be positive")
    remaining_minutes = remaining_seconds // 60
    timeout_minutes = max(1, math.ceil(reproduction_timeout_seconds / 60))
    return (
        f"You have {remaining_minutes} minutes remaining. The harness actually ran the "
        "current submission's reproduce.sh in a clean reproduction environment for up "
        f"to {timeout_minutes} minutes. Its actual execution output is saved at "
        f"{log_path}, and the run finished with exit status {reproduction_exit_code}. "
        "Use this result as evidence when reviewing the current submission, then "
        "continue the paper-replication task under the original instructions. Do not "
        "stop unless you have reproduced all core contributions of the paper. If they "
        "are already reproduced, verify reproduce.sh end to end and finish."
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

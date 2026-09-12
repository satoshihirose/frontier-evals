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
from paperbench.reproduction_policy import REPRODUCTION_SALVAGE_ATTEMPTS

DEFAULT_MIN_REMAINING_SECONDS = 60 * 60
QUICK_UNCHANGED_MAX_SECONDS = 5 * 60
MAX_CONSECUTIVE_QUICK_UNCHANGED = 2
COMPLETION_REVIEW_METADATA = "completion-review.json"
INITIAL_SUBMISSION_FILENAME = "initial-submission.tar.gz"
SUBMISSION_GIT_HEAD_COMMAND = f"git -C {SUBMISSION_DIR} rev-parse HEAD"
EXECUTION_FEEDBACK_DIR = "/home/logs/execution-feedback"
EXECUTION_FEEDBACK_SETUP_RESERVE_SECONDS = 60
EXECUTION_FEEDBACK_IMAGE = "pb-reproducer:latest"
EXECUTION_FEEDBACK_ENV_FILE = "/home/agent.env"
EXECUTION_FEEDBACK_NETWORK = "bridge"
EXECUTION_FEEDBACK_RETRY_THRESHOLD_SECONDS = 600
EXECUTION_FEEDBACK_ARTIFACT_RESERVE_SECONDS = 30
EXECUTION_FEEDBACK_MAX_ARTIFACT_BYTES = 10 * 1024 * 1024
EXECUTION_FEEDBACK_INPUT_VERSION = 2
EXECUTION_FEEDBACK_INFRASTRUCTURE_EXIT_CODE = 125
EXECUTION_FEEDBACK_INFRASTRUCTURE_MARKER = "/tmp/execution-feedback-infrastructure-error"
CompletionReviewMode = Literal["generic", "execution-log"]

_SNAPSHOT_SCRIPT = """
import hashlib
import json
import os
import stat
import sys

root, output, max_size = sys.argv[1], sys.argv[2], int(sys.argv[3])
excluded_directories = {
    ".cache",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".venv",
    "__pycache__",
    "node_modules",
    "venv",
}
snapshot = {}
for current, directories, files in os.walk(root, followlinks=False):
    directories[:] = sorted(name for name in directories if name not in excluded_directories)
    for name in sorted(files):
        path = os.path.join(current, name)
        info = os.lstat(path)
        if name.endswith(".pyc") or not stat.S_ISREG(info.st_mode) or info.st_size > max_size:
            continue
        relative = os.path.relpath(path, root)
        if relative in {"reproduce.log", "reproduce.log.creation_time"}:
            continue
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        snapshot[relative] = {
            "sha256": digest.hexdigest(),
            "mtime_ns": info.st_mtime_ns,
        }
with open(output, "w", encoding="utf-8") as handle:
    json.dump(snapshot, handle, sort_keys=True)
""".strip()

_COPY_CHANGED_ARTIFACTS_SCRIPT = """
import hashlib
import json
import os
import shutil
import stat
import sys

root, snapshot_path, destination, max_size = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
excluded_directories = {
    ".cache",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".venv",
    "__pycache__",
    "node_modules",
    "venv",
}
with open(snapshot_path, encoding="utf-8") as handle:
    before = json.load(handle)
os.makedirs(destination, exist_ok=True)
for current, directories, files in os.walk(root, followlinks=False):
    directories[:] = sorted(name for name in directories if name not in excluded_directories)
    for name in sorted(files):
        path = os.path.join(current, name)
        info = os.lstat(path)
        if name.endswith(".pyc") or not stat.S_ISREG(info.st_mode) or info.st_size > max_size:
            continue
        relative = os.path.relpath(path, root)
        if relative in {"reproduce.log", "reproduce.log.creation_time"}:
            continue
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        fingerprint = {
            "sha256": digest.hexdigest(),
            "mtime_ns": info.st_mtime_ns,
        }
        if before.get(relative) == fingerprint:
            continue
        target = os.path.join(destination, relative)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.copy2(path, target, follow_symlinks=False)
""".strip()


def remaining_budget_seconds(
    *, time_limit_seconds: int | float, start_time: int | float, now: int | float
) -> int:
    return max(0, math.floor(time_limit_seconds - (now - start_time)))


_COMPLETION_REVIEW_INSTRUCTION = (
    "Continue the current paper-replication task under the original instructions "
    "and use the remaining time to improve and verify your submission. Do not stop "
    "until you have reproduced all results from the paper. Before finishing, ensure "
    "the submission can reproduce all metrics, figures, tables, and findings from "
    "the paper. If all results are already reproduced, verify the submission against "
    "the original instructions and finish."
)


def build_completion_review_prompt(remaining_seconds: int) -> str:
    if remaining_seconds < 0:
        raise ValueError("remaining_seconds must be non-negative")
    remaining_minutes = remaining_seconds // 60
    return f"You have {remaining_minutes} minutes remaining. {_COMPLETION_REVIEW_INSTRUCTION}"


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


def is_execution_feedback_infrastructure_error(exit_code: int) -> bool:
    return exit_code == EXECUTION_FEEDBACK_INFRASTRUCTURE_EXIT_CODE


def execution_feedback_paths(iteration: int) -> tuple[str, str]:
    if iteration <= 0:
        raise ValueError("iteration must be positive")
    feedback_dir = f"{EXECUTION_FEEDBACK_DIR}/iteration-{iteration}"
    return (
        f"{feedback_dir}/reproduce.log",
        f"/tmp/paperbench-execution-feedback-{iteration}/harness.log",
    )


def _build_reproduction_container_command() -> str:
    return "\n".join(
        [
            "set -uo pipefail",
            "cd /submission",
            "rm -rf -- venv .venv",
            "mkdir -p /tmp/execution-feedback-artifacts",
            ": > reproduce.log",
            "setup_status=0",
            (
                'if [[ "$PB_USE_PY3_11" == 1 ]]; then '
                "update-alternatives --set python3 /usr/bin/python3.11 || setup_status=$?; fi"
            ),
            (
                'if [[ "$setup_status" == 0 && "$PB_MAKE_VENV" == 1 ]]; then '
                "python3 -m venv venv && source venv/bin/activate || setup_status=$?; fi"
            ),
            (
                'if [[ "$setup_status" != 0 ]]; then '
                "tar -czf /tmp/execution-feedback-artifacts.tar.gz "
                "-C /tmp/execution-feedback-artifacts .; exit \"$setup_status\"; fi"
            ),
            (
                f"python3 -c {shlex.quote(_SNAPSHOT_SCRIPT)} /submission "
                f"/tmp/execution-feedback-before.json {EXECUTION_FEEDBACK_MAX_ARTIFACT_BYTES} "
                f"|| {{ touch {EXECUTION_FEEDBACK_INFRASTRUCTURE_MARKER}; "
                f"exit {EXECUTION_FEEDBACK_INFRASTRUCTURE_EXIT_CODE}; }}"
            ),
            "date +%s > reproduce.log.creation_time",
            (
                f"attempt_timeout=$((PB_DIAGNOSTIC_DEADLINE - $(date +%s) - "
                f"{EXECUTION_FEEDBACK_ARTIFACT_RESERVE_SECONDS}))"
            ),
            (
                "if (( attempt_timeout <= 0 )); then "
                "tar -czf /tmp/execution-feedback-artifacts.tar.gz "
                "-C /tmp/execution-feedback-artifacts .; exit 124; fi"
            ),
            "set +e",
            (
                'timeout --signal=TERM --kill-after=30s "${attempt_timeout}s" '
                "bash -o pipefail -c 'bash reproduce.sh 2>&1 | tee reproduce.log'"
            ),
            "reproduction_status=$?",
            "set -e",
            (
                f"python3 -c {shlex.quote(_COPY_CHANGED_ARTIFACTS_SCRIPT)} /submission "
                "/tmp/execution-feedback-before.json /tmp/execution-feedback-artifacts "
                f"{EXECUTION_FEEDBACK_MAX_ARTIFACT_BYTES} "
                f"|| {{ touch {EXECUTION_FEEDBACK_INFRASTRUCTURE_MARKER}; "
                f"exit {EXECUTION_FEEDBACK_INFRASTRUCTURE_EXIT_CODE}; }}"
            ),
            (
                "tar -czf /tmp/execution-feedback-artifacts.tar.gz "
                "-C /tmp/execution-feedback-artifacts . "
                f"|| {{ touch {EXECUTION_FEEDBACK_INFRASTRUCTURE_MARKER}; "
                f"exit {EXECUTION_FEEDBACK_INFRASTRUCTURE_EXIT_CODE}; }}"
            ),
            'exit "$reproduction_status"',
        ]
    )


def _build_execution_feedback_driver(*, iteration: int, timeout_seconds: int) -> str:
    log_path, harness_log_path = execution_feedback_paths(iteration)
    feedback_dir = log_path.rsplit("/", 1)[0]
    harness_dir = harness_log_path.rsplit("/", 1)[0]
    container_prefix = f"pb-execution-feedback-{uuid.uuid4().hex}"
    attempt_options = " ".join(
        shlex.quote(f"{int(attempt.use_py3_11)} {int(attempt.make_venv)}")
        for attempt in REPRODUCTION_SALVAGE_ATTEMPTS
    )
    container_command = _build_reproduction_container_command()
    driver = "\n".join(
        [
            "set -uo pipefail",
            f"feedback_dir={shlex.quote(feedback_dir)}",
            f"harness_dir={shlex.quote(harness_dir)}",
            f"harness_log={shlex.quote(harness_log_path)}",
            'attempts_dir="$harness_dir/attempts"',
            f"container_prefix={shlex.quote(container_prefix)}",
            'container_name=""',
            (
                'cleanup() { if [[ -n "$container_name" ]]; then '
                'docker rm -f "$container_name" >/dev/null 2>&1 || true; fi; }'
            ),
            "trap cleanup EXIT",
            f"diagnostic_budget_seconds={timeout_seconds}",
            'deadline_epoch=$(($(date +%s) + diagnostic_budget_seconds))',
            f"attempt_options=({attempt_options})",
            'rm -rf -- "$feedback_dir" "$harness_dir"',
            'mkdir -p "$feedback_dir" "$attempts_dir"',
            ': > "$harness_log"',
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
            "env_args=()",
            (
                f"if [[ -f {shlex.quote(EXECUTION_FEEDBACK_ENV_FILE)} ]]; then "
                f"env_args=(--env-file {shlex.quote(EXECUTION_FEEDBACK_ENV_FILE)}); fi"
            ),
            "durations=()",
            "statuses=()",
            "attempt_count=0",
            "for option in \"${attempt_options[@]}\"; do",
            (
                f"  remaining=$((deadline_epoch - $(date +%s) - "
                f"{EXECUTION_FEEDBACK_ARTIFACT_RESERVE_SECONDS}))"
            ),
            "  if (( remaining <= 0 )); then break; fi",
            '  read -r use_py3_11 make_venv <<< "$option"',
            "  attempt_index=$attempt_count",
            "  attempt_number=$((attempt_index + 1))",
            '  attempt_dir="$attempts_dir/attempt-$attempt_number"',
            '  container_name="$container_prefix-$attempt_number"',
            '  mkdir -p "$attempt_dir/artifacts"',
            '  started=$SECONDS',
            '  docker rm -f "$container_name" >/dev/null 2>&1 || true',
            (
                '  if ! docker create --name "$container_name" --shm-size 8g '
                f"--network {shlex.quote(EXECUTION_FEEDBACK_NETWORK)} "
                '"${gpu_args[@]}" "${env_args[@]}" '
                '-e "PB_USE_PY3_11=$use_py3_11" -e "PB_MAKE_VENV=$make_venv" '
                '-e "PB_DIAGNOSTIC_DEADLINE=$deadline_epoch" '
                f"{shlex.quote(EXECUTION_FEEDBACK_IMAGE)} bash -lc "
                f"{shlex.quote(container_command)} >> \"$harness_log\" 2>&1; then"
            ),
            f"    exit {EXECUTION_FEEDBACK_INFRASTRUCTURE_EXIT_CODE}",
            "  fi",
            (
                f"  if ! docker cp {shlex.quote(SUBMISSION_DIR)}/. "
                '"$container_name:/submission" >> "$harness_log" 2>&1; then'
            ),
            '    docker rm -f "$container_name" >/dev/null 2>&1 || true',
            f"    exit {EXECUTION_FEEDBACK_INFRASTRUCTURE_EXIT_CODE}",
            "  fi",
            (
                f"  remaining=$((deadline_epoch - $(date +%s) - "
                f"{EXECUTION_FEEDBACK_ARTIFACT_RESERVE_SECONDS}))"
            ),
            "  if (( remaining <= 0 )); then",
            '    docker rm -f "$container_name" >/dev/null 2>&1 || true',
            "    break",
            "  fi",
            '  docker start -a "$container_name" >> "$harness_log" 2>&1',
            "  attempt_status=$?",
            "  duration=$((SECONDS - started))",
            '  durations+=("$duration")',
            '  statuses+=("$attempt_status")',
            "  attempt_count=$((attempt_count + 1))",
            (
                f"  if docker cp \"$container_name:{EXECUTION_FEEDBACK_INFRASTRUCTURE_MARKER}\" "
                '"$harness_dir/infrastructure-error" >> "$harness_log" 2>&1; then'
            ),
            '    docker rm -f "$container_name" >/dev/null 2>&1 || true',
            f"    exit {EXECUTION_FEEDBACK_INFRASTRUCTURE_EXIT_CODE}",
            "  fi",
            (
                '  if ! docker cp "$container_name:/submission/reproduce.log" '
                '"$attempt_dir/reproduce.log" >> "$harness_log" 2>&1; then'
            ),
            '    docker rm -f "$container_name" >/dev/null 2>&1 || true',
            f"    exit {EXECUTION_FEEDBACK_INFRASTRUCTURE_EXIT_CODE}",
            "  fi",
            (
                '  if ! docker cp "$container_name:/tmp/execution-feedback-artifacts.tar.gz" '
                '"$attempt_dir/artifacts.tar.gz" >> "$harness_log" 2>&1; then'
            ),
            '    docker rm -f "$container_name" >/dev/null 2>&1 || true',
            f"    exit {EXECUTION_FEEDBACK_INFRASTRUCTURE_EXIT_CODE}",
            "  fi",
            '  tar -xzf "$attempt_dir/artifacts.tar.gz" -C "$attempt_dir/artifacts"',
            '  rm -f -- "$attempt_dir/artifacts.tar.gz"',
            '  docker rm -f "$container_name" >/dev/null 2>&1 || true',
            (
                f"  if [[ \"$attempt_status\" == 124 || \"$attempt_status\" == 137 "
                f"|| ( \"$attempt_status\" == 0 "
                f"&& \"$duration\" -ge {EXECUTION_FEEDBACK_RETRY_THRESHOLD_SECONDS} ) ]]; then"
            ),
            "    break",
            "  fi",
            "done",
            "if (( attempt_count == 0 )); then",
            f"  exit {EXECUTION_FEEDBACK_INFRASTRUCTURE_EXIT_CODE}",
            "fi",
            "selected_attempt=-1",
            "for ((index=0; index<attempt_count; index++)); do",
            "  if (( statuses[index] == 0 )) && {",
            "    (( selected_attempt < 0 )) || (( durations[index] > durations[selected_attempt] ))",
            "  }; then",
            "    selected_attempt=$index",
            "  fi",
            "done",
            "if (( selected_attempt < 0 )); then",
            "  selected_attempt=0",
            "  for ((index=1; index<attempt_count; index++)); do",
            "    if (( durations[index] > durations[selected_attempt] )); then",
            "      selected_attempt=$index",
            "    fi",
            "  done",
            "fi",
            "selected_number=$((selected_attempt + 1))",
            'cp "$attempts_dir/attempt-$selected_number/reproduce.log" '
            '"$feedback_dir/reproduce.log"',
            'mkdir -p "$feedback_dir/artifacts"',
            'cp -a "$attempts_dir/attempt-$selected_number/artifacts/." '
            '"$feedback_dir/artifacts/"',
            'selected_status="${statuses[selected_attempt]}"',
            'rm -rf -- "$harness_dir"',
            'cat "$feedback_dir/reproduce.log"',
            'exit "$selected_status"',
        ]
    )
    return f"bash -lc {shlex.quote(driver)}"


def build_execution_feedback_command(*, iteration: int, timeout_seconds: int) -> str:
    """Run formal-style salvage attempts in disposable reproduction containers."""
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    return _build_execution_feedback_driver(iteration=iteration, timeout_seconds=timeout_seconds)


def build_execution_feedback_prompt(
    *,
    remaining_seconds: int,
    log_path: str,
    reproduction_timeout_seconds: int,
) -> str:
    if remaining_seconds < 0:
        raise ValueError("remaining_seconds must be non-negative")
    if reproduction_timeout_seconds <= 0:
        raise ValueError("reproduction_timeout_seconds must be positive")
    remaining_minutes = remaining_seconds // 60
    timeout_minutes = max(1, math.ceil(reproduction_timeout_seconds / 60))
    artifacts_path = f"{log_path.rsplit('/', 1)[0]}/artifacts"
    return (
        f"You have {remaining_minutes} minutes remaining. A diagnostic reproduction "
        f"was run for up to {timeout_minutes} minutes. Review the execution log at "
        f"{log_path} and any relevant generated artifacts under {artifacts_path}. "
        f"{_COMPLETION_REVIEW_INSTRUCTION}"
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

from __future__ import annotations

import errno
from dataclasses import replace
from pathlib import Path

import pytest
from paperbench import reproduce as reproduce_module
from paperbench.nano.structs import ReproductionMetadata


def test_materialize_canonical_submission_uses_atomic_hardlink(tmp_path: Path) -> None:
    source = tmp_path / "submission_executed_attempt_1.tar.gz"
    destination = tmp_path / "submission_executed.tar.gz"
    source.write_bytes(b"selected-attempt")
    destination.write_bytes(b"stale-canonical")

    method = reproduce_module._materialize_canonical_submission(str(source), str(destination))

    assert method == "hardlink"
    assert destination.read_bytes() == b"selected-attempt"
    assert source.stat().st_ino == destination.stat().st_ino


def test_materialize_canonical_submission_falls_back_to_blob_copy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "submission_executed_attempt_1.tar.gz"
    destination = tmp_path / "submission_executed.tar.gz"
    source.write_bytes(b"selected-attempt")
    copied: list[tuple[str, str, bool]] = []

    def unavailable_link(*_args: object, **_kwargs: object) -> None:
        raise OSError(errno.EXDEV, "cross-device link")

    def fake_copy(source_path: str, destination_path: str, *, overwrite: bool) -> None:
        copied.append((source_path, destination_path, overwrite))

    monkeypatch.setattr(reproduce_module.os, "link", unavailable_link)
    monkeypatch.setattr(reproduce_module.bf, "copy", fake_copy)

    method = reproduce_module._materialize_canonical_submission(str(source), str(destination))

    assert method == "copy"
    assert copied == [(str(source), str(destination), True)]


def _metadata(seconds: float, attempt_index: int, *, exit_code: int = 0) -> ReproductionMetadata:
    return ReproductionMetadata(
        is_valid_git_repo=True,
        git_log="git log",
        repro_script_exists=True,
        files_before_reproduce="before",
        files_after_reproduce="after",
        timedout=False,
        repro_log=f"attempt {attempt_index}",
        repro_exit_code=exit_code,
        repro_execution_time=seconds,
        git_status_after_reproduce="clean",
        executed_submission=f"attempt-{attempt_index}.tar.gz",
    )


@pytest.mark.asyncio
async def test_all_short_attempts_promote_the_longest_attempt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    attempt_times = iter([100.0, 350.0, 200.0, 150.0])
    attempts: list[int | None] = []
    copied: list[tuple[str, str, bool]] = []
    written: list[str] = []

    async def fake_reproduce_on_computer(*, attempt_index: int | None, **kwargs: object):
        attempts.append(attempt_index)
        return _metadata(next(attempt_times), int(attempt_index or 0))

    def fake_copy(source: str, destination: str, *, overwrite: bool) -> None:
        copied.append((source, destination, overwrite))

    monkeypatch.setattr(reproduce_module, "reproduce_on_computer", fake_reproduce_on_computer)
    monkeypatch.setattr(reproduce_module.bf, "copy", fake_copy)
    monkeypatch.setattr(
        reproduce_module.bf,
        "write_bytes",
        lambda path, data: written.append(path),
    )

    submission = str(tmp_path / "submission.tar.gz")
    result = await reproduce_module.reproduce_on_computer_with_salvaging(
        computer_runtime=object(),  # type: ignore[arg-type]
        computer_config=object(),  # type: ignore[arg-type]
        runtime_config=object(),  # type: ignore[arg-type]
        submission_path=submission,
        run_group_id="group",
        runs_dir=str(tmp_path),
        run_id="run",
        run_dir=str(tmp_path),
        retry_threshold=600,
    )

    assert attempts == [0, 1, 2, 3]
    assert result.repro_execution_time == 350.0
    assert result.executed_submission == submission.replace(".tar.gz", "_executed.tar.gz")
    assert len(result.retried_results) == 3
    assert written == [submission.replace(".tar.gz", "_executed_metadata.json")]
    assert copied == [
        (
            "attempt-1.tar.gz",
            submission.replace(".tar.gz", "_executed.tar.gz"),
            True,
        )
    ]


@pytest.mark.asyncio
async def test_threshold_qualifying_attempt_still_stops_and_is_selected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    attempt_times = iter([100.0, 650.0, 700.0, 800.0])
    attempts: list[int | None] = []
    copied: list[tuple[str, str, bool]] = []

    async def fake_reproduce_on_computer(*, attempt_index: int | None, **kwargs: object):
        attempts.append(attempt_index)
        return _metadata(next(attempt_times), int(attempt_index or 0))

    def fake_copy(source: str, destination: str, *, overwrite: bool) -> None:
        copied.append((source, destination, overwrite))

    monkeypatch.setattr(reproduce_module, "reproduce_on_computer", fake_reproduce_on_computer)
    monkeypatch.setattr(reproduce_module.bf, "copy", fake_copy)
    monkeypatch.setattr(reproduce_module.bf, "write_bytes", lambda *args, **kwargs: None)

    submission = str(tmp_path / "submission.tar.gz")
    result = await reproduce_module.reproduce_on_computer_with_salvaging(
        computer_runtime=object(),  # type: ignore[arg-type]
        computer_config=object(),  # type: ignore[arg-type]
        runtime_config=object(),  # type: ignore[arg-type]
        submission_path=submission,
        run_group_id="group",
        runs_dir=str(tmp_path),
        run_id="run",
        run_dir=str(tmp_path),
        retry_threshold=600,
    )

    assert attempts == [0, 1]
    assert result.repro_execution_time == 650.0
    assert len(result.retried_results) == 1
    assert copied == [
        (
            "attempt-1.tar.gz",
            submission.replace(".tar.gz", "_executed.tar.gz"),
            True,
        )
    ]


@pytest.mark.asyncio
async def test_timed_out_attempt_is_terminal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    attempts: list[int | None] = []

    async def fake_reproduce_on_computer(*, attempt_index: int | None, **kwargs: object):
        attempts.append(attempt_index)
        metadata = _metadata(700.0, int(attempt_index or 0))
        return replace(metadata, timedout=attempt_index == 0)

    monkeypatch.setattr(reproduce_module, "reproduce_on_computer", fake_reproduce_on_computer)
    monkeypatch.setattr(reproduce_module.bf, "copy", lambda *args, **kwargs: None)
    monkeypatch.setattr(reproduce_module.bf, "write_bytes", lambda *args, **kwargs: None)

    result = await reproduce_module.reproduce_on_computer_with_salvaging(
        computer_runtime=object(),  # type: ignore[arg-type]
        computer_config=object(),  # type: ignore[arg-type]
        runtime_config=object(),  # type: ignore[arg-type]
        submission_path=str(tmp_path / "submission.tar.gz"),
        run_group_id="group",
        runs_dir=str(tmp_path),
        run_id="run",
        run_dir=str(tmp_path),
        retry_threshold=600,
    )

    assert attempts == [0]
    assert result.timedout is True
    assert result.retried_results == []


@pytest.mark.asyncio
async def test_threshold_qualifying_failed_attempt_is_retried(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    attempt_outcomes = iter([(100.0, 0), (650.0, 1), (700.0, 0)])
    attempts: list[int | None] = []

    async def fake_reproduce_on_computer(*, attempt_index: int | None, **kwargs: object):
        attempts.append(attempt_index)
        seconds, exit_code = next(attempt_outcomes)
        return _metadata(seconds, int(attempt_index or 0), exit_code=exit_code)

    monkeypatch.setattr(reproduce_module, "reproduce_on_computer", fake_reproduce_on_computer)
    monkeypatch.setattr(reproduce_module.bf, "copy", lambda *args, **kwargs: None)
    monkeypatch.setattr(reproduce_module.bf, "write_bytes", lambda *args, **kwargs: None)

    result = await reproduce_module.reproduce_on_computer_with_salvaging(
        computer_runtime=object(),  # type: ignore[arg-type]
        computer_config=object(),  # type: ignore[arg-type]
        runtime_config=object(),  # type: ignore[arg-type]
        submission_path=str(tmp_path / "submission.tar.gz"),
        run_group_id="group",
        runs_dir=str(tmp_path),
        run_id="run",
        run_dir=str(tmp_path),
        retry_threshold=600,
    )

    assert attempts == [0, 1, 2]
    assert result.repro_execution_time == 700.0
    assert result.repro_exit_code == 0


@pytest.mark.asyncio
async def test_longest_successful_attempt_wins_over_longer_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    attempt_outcomes = iter([(100.0, 1), (350.0, 1), (200.0, 0), (150.0, 1)])

    async def fake_reproduce_on_computer(*, attempt_index: int | None, **kwargs: object):
        seconds, exit_code = next(attempt_outcomes)
        return _metadata(seconds, int(attempt_index or 0), exit_code=exit_code)

    copied: list[tuple[str, str, bool]] = []
    monkeypatch.setattr(reproduce_module, "reproduce_on_computer", fake_reproduce_on_computer)
    monkeypatch.setattr(
        reproduce_module.bf,
        "copy",
        lambda source, destination, *, overwrite: copied.append((source, destination, overwrite)),
    )
    monkeypatch.setattr(reproduce_module.bf, "write_bytes", lambda *args, **kwargs: None)

    submission = str(tmp_path / "submission.tar.gz")
    result = await reproduce_module.reproduce_on_computer_with_salvaging(
        computer_runtime=object(),  # type: ignore[arg-type]
        computer_config=object(),  # type: ignore[arg-type]
        runtime_config=object(),  # type: ignore[arg-type]
        submission_path=submission,
        run_group_id="group",
        runs_dir=str(tmp_path),
        run_id="run",
        run_dir=str(tmp_path),
        retry_threshold=600,
    )

    assert result.repro_exit_code == 0
    assert result.repro_execution_time == 200.0
    assert copied == [
        (
            "attempt-2.tar.gz",
            submission.replace(".tar.gz", "_executed.tar.gz"),
            True,
        )
    ]

from __future__ import annotations

import tarfile
from pathlib import Path
from typing import Any

import httpx
import openai
import pytest

from paperbench import grade as grade_module
from paperbench.grade import grade_submission, hash_submission_tree
from paperbench.judge.base import Judge
from paperbench.judge.codex_cli_turn_completer import CodexCliTurnCompleter
from paperbench.judge.graded_task_node import GradedTaskNode
from paperbench.judge.leaf_checkpoint import LeafCheckpointStore
from paperbench.rubric.tasks import TaskNode
from paperbench.scripts import run_judge as run_judge_script


def _leaf(node_id: str, requirements: str) -> TaskNode:
    return TaskNode(
        id=node_id,
        requirements=requirements,
        weight=1,
        task_category="Code Development",
    )


def _rubric(first_requirement: str = "Implement A.") -> TaskNode:
    return TaskNode(
        id="root",
        requirements="Root",
        weight=1,
        sub_tasks=[
            _leaf("leaf-a", first_requirement),
            _leaf("leaf-b", "Implement B."),
        ],
    )


class RecordingJudge(Judge):
    def __init__(
        self,
        *,
        paper_path: Path,
        rubric: TaskNode,
        submission_dir: Path,
        leaf_checkpoint_store: LeafCheckpointStore,
        fail_ids: set[str] | None = None,
        failures_before_success: dict[str, int] | None = None,
        invalid_results_before_success: dict[str, int] | None = None,
    ) -> None:
        super().__init__(
            paper_path=paper_path,
            rubric=rubric,
            addendum=None,
            judge_addendum=None,
            submission_dir=submission_dir,
            leaf_checkpoint_store=leaf_checkpoint_store,
        )
        self.fail_ids = fail_ids or set()
        self.failures_before_success = failures_before_success or {}
        self.invalid_results_before_success = invalid_results_before_success or {}
        self.calls: list[str] = []

    @property
    def judge_type(self) -> str:
        return "recording"

    async def grade_leaf(self, task: TaskNode) -> GradedTaskNode:
        self.calls.append(task.id)
        if task.id in self.fail_ids:
            raise RuntimeError("interrupted leaf")
        remaining_failures = self.failures_before_success.get(task.id, 0)
        if self.calls.count(task.id) <= remaining_failures:
            raise RuntimeError("transient leaf failure")
        remaining_invalid_results = self.invalid_results_before_success.get(task.id, 0)
        if self.calls.count(task.id) <= remaining_invalid_results:
            return GradedTaskNode.from_task(
                task,
                score=0.0,
                valid_score=False,
                explanation="transient invalid result",
            )
        return GradedTaskNode.from_task(
            task,
            score=1.0,
            valid_score=True,
            explanation="graded",
        )

    async def grade_subtree(self, task: TaskNode) -> GradedTaskNode:
        raise AssertionError(f"Unexpected subtree grading for {task.id}")


def _judge(
    tmp_path: Path,
    rubric: TaskNode,
    *,
    context_hash: str,
    fail_ids: set[str] | None = None,
    failures_before_success: dict[str, int] | None = None,
    invalid_results_before_success: dict[str, int] | None = None,
) -> RecordingJudge:
    submission = tmp_path / "submission"
    submission.mkdir(exist_ok=True)
    return RecordingJudge(
        paper_path=tmp_path / "paper.pdf",
        rubric=rubric,
        submission_dir=submission,
        leaf_checkpoint_store=LeafCheckpointStore(
            path=str(tmp_path / "leaf-checkpoints.jsonl"),
            context_hash=context_hash,
        ),
        fail_ids=fail_ids,
        failures_before_success=failures_before_success,
        invalid_results_before_success=invalid_results_before_success,
    )


@pytest.mark.asyncio
async def test_judge_resumes_only_unfinished_leaves(tmp_path: Path) -> None:
    first = _judge(tmp_path, _rubric(), context_hash="same-run", fail_ids={"leaf-b"})

    first_result = await first.judge()

    assert set(first.calls) == {"leaf-a", "leaf-b"}
    assert first_result.find("leaf-a").valid_score is True
    assert first_result.find("leaf-b").valid_score is False

    resumed = _judge(tmp_path, _rubric(), context_hash="same-run")
    resumed_result = await resumed.judge()

    assert resumed.calls == ["leaf-b"]
    assert resumed_result.find("leaf-a").explanation == "graded"
    assert resumed_result.score == 1.0


@pytest.mark.asyncio
async def test_judge_retries_a_transient_invalid_leaf_up_to_three_times(
    tmp_path: Path,
) -> None:
    judge = _judge(
        tmp_path,
        _rubric(),
        context_hash="retry-transient-leaf",
        invalid_results_before_success={"leaf-b": 3},
    )

    result = await judge.judge()

    assert judge.calls.count("leaf-a") == 1
    assert judge.calls.count("leaf-b") == 4
    assert result.find("leaf-b").valid_score is True


@pytest.mark.asyncio
async def test_judge_returns_invalid_after_three_leaf_retries(tmp_path: Path) -> None:
    judge = _judge(
        tmp_path,
        _rubric(),
        context_hash="retry-persistent-leaf",
        fail_ids={"leaf-b"},
    )

    result = await judge.judge()

    assert judge.calls.count("leaf-a") == 1
    assert judge.calls.count("leaf-b") == 4
    assert result.find("leaf-b").valid_score is False


@pytest.mark.asyncio
async def test_judge_records_timeout_as_invalid_without_repeating_leaf(
    tmp_path: Path,
) -> None:
    judge = _judge(tmp_path, _rubric(), context_hash="timeout-leaf")

    async def grade_leaf(task: TaskNode) -> GradedTaskNode:
        judge.calls.append(task.id)
        if task.id == "leaf-b":
            raise openai.APITimeoutError(request=httpx.Request("POST", "http://judge"))
        return GradedTaskNode.from_task(
            task,
            score=1.0,
            valid_score=True,
            explanation="graded",
        )

    result = await judge.judge(grade_leaf_fn=grade_leaf)

    assert judge.calls.count("leaf-a") == 1
    assert judge.calls.count("leaf-b") == 1
    assert result.find("leaf-b").valid_score is False
    assert result.score == 0.5


@pytest.mark.asyncio
async def test_judge_does_not_reuse_checkpoint_for_a_different_context(tmp_path: Path) -> None:
    first = _judge(tmp_path, _rubric(), context_hash="first-run")
    await first.judge()

    second = _judge(tmp_path, _rubric(), context_hash="different-run")
    await second.judge()

    assert set(second.calls) == {"leaf-a", "leaf-b"}


@pytest.mark.asyncio
async def test_judge_does_not_reuse_changed_leaf_with_the_same_id(tmp_path: Path) -> None:
    first = _judge(tmp_path, _rubric(), context_hash="same-run")
    await first.judge()

    second = _judge(
        tmp_path,
        _rubric(first_requirement="Implement the revised A."),
        context_hash="same-run",
    )
    await second.judge()

    assert second.calls == ["leaf-a"]


@pytest.mark.asyncio
async def test_grade_submission_places_checkpoint_beside_grader_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    submission_dir = tmp_path / "submission"
    submission_dir.mkdir()
    (submission_dir / "solution.py").write_text("pass\n", encoding="utf-8")
    submission_archive = tmp_path / "submission.tar.gz"
    with tarfile.open(submission_archive, "w:gz") as archive:
        archive.add(submission_dir / "solution.py", arcname="solution.py")
    grader_output = tmp_path / "submission_grader_output_0.json"
    captured: dict[str, Any] = {}
    graded_leaf = GradedTaskNode.from_task(
        _leaf("leaf", "Implement it."),
        score=1.0,
        valid_score=True,
        explanation="graded",
    )

    async def fake_run_judge(**kwargs: Any) -> GradedTaskNode:
        captured.update(kwargs)
        return graded_leaf

    monkeypatch.setattr(grade_module, "run_judge", fake_run_judge)

    output = await grade_submission(
        submission_path=str(submission_archive),
        grader_upload_path=str(grader_output),
        paper_id="test-paper",
        judge_type="dummy",
        completer_config=CodexCliTurnCompleter.Config(),
        run_group_id="group",
        runs_dir=str(tmp_path),
        run_id="run",
    )

    assert output is not None
    assert captured["leaf_checkpoint_path"] == str(
        tmp_path / "submission_grader_output_0_leaf_checkpoints.jsonl"
    )
    assert captured["submission_fingerprint"] == hash_submission_tree(submission_dir)


@pytest.mark.asyncio
async def test_standalone_judge_uses_explicit_checkpoint_and_submission_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    submission_dir = tmp_path / "submission"
    submission_dir.mkdir()
    (submission_dir / "solution.py").write_text("pass\n", encoding="utf-8")
    out_dir = tmp_path / "judge-output"
    checkpoint_path = out_dir / "leaf-checkpoints.jsonl"
    captured: dict[str, Any] = {}
    graded_leaf = GradedTaskNode.from_task(
        _leaf("leaf", "Implement it."),
        score=1.0,
        valid_score=True,
        explanation="graded",
    )

    async def fake_run_judge(**kwargs: Any) -> GradedTaskNode:
        captured.update(kwargs)
        return graded_leaf

    monkeypatch.setattr(run_judge_script, "run_judge", fake_run_judge)

    await run_judge_script.main(
        submission_path=submission_dir,
        paper_id="test-paper",
        judge_type="dummy",
        max_depth=999,
        out_dir=out_dir,
        code_only=False,
        leaf_checkpoint_path=checkpoint_path,
    )

    assert captured["leaf_checkpoint_path"] == str(checkpoint_path)
    assert captured["submission_fingerprint"] == hash_submission_tree(submission_dir)
    assert (out_dir / "grader_output.json").exists()

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import paperbench.solvers.basicagent.solver as basicagent_solver_module
from paperbench.solvers.basicagent.solver import BasicAgentSolver
from paperbench.solvers.basicagent.utils import get_task_instruction_text


def test_basicagent_uses_the_task_prompt_as_its_common_contract() -> None:
    prompt = (
        "Common PaperBench contract. Do not rely on hardcoded absolute paths.\n\n"
        "ADDITIONAL REPRODUCTION REQUIREMENTS\nRead every row.\n"
    )
    task = SimpleNamespace(run_id="test-run", prompt=[{"role": "user", "content": prompt}])

    assert get_task_instruction_text(task) == prompt


@pytest.mark.parametrize(
    "prompt",
    [
        [],
        [{"role": "user"}],
        [{"role": "user", "content": 123}],
    ],
)
def test_basicagent_rejects_an_invalid_task_prompt(prompt: object) -> None:
    task = SimpleNamespace(run_id="test-run", prompt=prompt)

    with pytest.raises((TypeError, ValueError)):
        get_task_instruction_text(task)


def test_completion_review_is_opt_in_with_a_fixed_one_hour_floor() -> None:
    solver = BasicAgentSolver()

    assert solver.completion_review is False
    assert solver.completion_review_min_remaining_seconds == 3600


@pytest.mark.asyncio
async def test_basicagent_repeats_review_while_at_least_one_hour_remains(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = 0

    async def fake_request(**_: object) -> SimpleNamespace:
        nonlocal calls
        calls += 1
        return SimpleNamespace(
            response_messages=[],
            tool_calls=[SimpleNamespace(call_id=f"submit-{calls}")],
            time_spent_retrying=0,
        )

    async def fake_handle(*_: object, **__: object) -> None:
        return None

    async def fake_optional_upload(**kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            last_time_uploaded=kwargs["last_time_uploaded"],
            upload_task=None,
        )

    async def fake_upload(**_: object) -> None:
        return None

    remaining = iter([5000, 4000, 3599])
    monkeypatch.setattr(basicagent_solver_module, "make_completer_request", fake_request)
    monkeypatch.setattr(basicagent_solver_module, "handle_tool_call", fake_handle)
    monkeypatch.setattr(
        basicagent_solver_module,
        "optionally_upload_heavy_logs",
        fake_optional_upload,
    )
    monkeypatch.setattr(basicagent_solver_module, "upload_heavy_logs", fake_upload)
    monkeypatch.setattr(
        basicagent_solver_module,
        "remaining_budget_seconds",
        lambda **_: next(remaining),
    )
    monkeypatch.setattr(
        basicagent_solver_module,
        "snapshot_initial_submission",
        lambda _: "/tmp/initial-submission.tar.gz",
    )

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    task = SimpleNamespace(
        run_id="test-run",
        run_group_id="group",
        runs_dir=str(tmp_path),
        run_dir=str(run_dir),
    )
    solver = BasicAgentSolver(time_limit=7200, completion_review=True)
    monkeypatch.setattr(type(solver.completer_config), "build", lambda _: object())
    monkeypatch.setattr(BasicAgentSolver, "_has_agent_finished", lambda *_args, **_kwargs: False)

    steps = await solver._execute_agent(
        computer=SimpleNamespace(),
        task=task,
        prompt=[{"role": "user", "content": "task"}],
        start_time=0,
    )

    assert steps == 3
    assert calls == 3
    completion = json.loads((run_dir / "completion-review.json").read_text())
    assert completion["performed"] is True
    assert completion["review_count"] == 2
    assert [item["remaining_seconds_at_start"] for item in completion["iterations"]] == [
        5000,
        4000,
    ]
    assert completion["completion_reason"] == "insufficient-original-budget"


@pytest.mark.asyncio
async def test_basicagent_stops_after_two_quick_unchanged_reviews(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = 0

    async def fake_request(**_: object) -> SimpleNamespace:
        nonlocal calls
        calls += 1
        return SimpleNamespace(
            response_messages=[],
            tool_calls=[SimpleNamespace(call_id=f"submit-{calls}")],
            time_spent_retrying=0,
        )

    async def fake_handle(*_: object, **__: object) -> None:
        return None

    async def fake_optional_upload(**kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            last_time_uploaded=kwargs["last_time_uploaded"],
            upload_task=None,
        )

    async def fake_upload(**_: object) -> None:
        return None

    heads = iter(["b" * 40] * 4)

    async def fake_submission_git_head(*_: object, **__: object) -> str:
        return next(heads)

    remaining = iter([5000, 4900, 4800, 3599])
    monkeypatch.setattr(basicagent_solver_module, "make_completer_request", fake_request)
    monkeypatch.setattr(basicagent_solver_module, "handle_tool_call", fake_handle)
    monkeypatch.setattr(
        basicagent_solver_module, "optionally_upload_heavy_logs", fake_optional_upload
    )
    monkeypatch.setattr(basicagent_solver_module, "upload_heavy_logs", fake_upload)
    monkeypatch.setattr(
        basicagent_solver_module,
        "remaining_budget_seconds",
        lambda **_: next(remaining),
    )
    monkeypatch.setattr(
        basicagent_solver_module,
        "snapshot_initial_submission",
        lambda _: "/tmp/initial-submission.tar.gz",
    )
    monkeypatch.setattr(
        basicagent_solver_module,
        "get_submission_git_head",
        fake_submission_git_head,
        raising=False,
    )

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    task = SimpleNamespace(
        run_id="test-run",
        run_group_id="group",
        runs_dir=str(tmp_path),
        run_dir=str(run_dir),
    )
    solver = BasicAgentSolver(time_limit=7200, completion_review=True)
    monkeypatch.setattr(type(solver.completer_config), "build", lambda _: object())
    monkeypatch.setattr(BasicAgentSolver, "_has_agent_finished", lambda *_args, **_kwargs: False)

    steps = await solver._execute_agent(
        computer=SimpleNamespace(),
        task=task,
        prompt=[{"role": "user", "content": "task"}],
        start_time=0,
    )

    assert steps == 3
    assert calls == 3
    completion = json.loads((run_dir / "completion-review.json").read_text())
    assert completion["review_count"] == 2
    assert completion["completion_reason"] == "consecutive-quick-unchanged"
    assert completion["consecutive_quick_unchanged"] == 2
    assert all(item["quick_unchanged"] is True for item in completion["iterations"])

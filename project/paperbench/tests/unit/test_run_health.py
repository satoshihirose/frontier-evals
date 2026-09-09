from paperbench.nano.eval import build_run_health
from paperbench.nano.structs import PaperBenchResult


def _result(*, has_reproduction: bool, has_judge: bool) -> PaperBenchResult:
    result = PaperBenchResult(
        paper_id="paper",
        run_id="paper_run",
        submission_exists=True,
        skipped_reproduction=False,
        code_only=False,
        resources_provided=False,
    )
    result.agent_output = object()  # type: ignore[assignment]
    if has_reproduction:
        result.reproduction_metadata = object()  # type: ignore[assignment]
    if has_judge:
        result.judge_output = object()  # type: ignore[assignment]
    return result


def test_run_health_omits_failure_counts_when_grading_is_disabled() -> None:
    health = build_run_health(
        [_result(has_reproduction=False, has_judge=False)],
        grading_enabled=False,
    )

    assert health == {
        "mode": "reproduction_only",
        "n_rollouts_failed": 0,
        "n_reproductions_failed": None,
        "n_gradings_failed": None,
    }


def test_run_health_keeps_failure_counts_for_normal_evaluation() -> None:
    health = build_run_health(
        [_result(has_reproduction=False, has_judge=False)],
        grading_enabled=True,
    )

    assert health == {
        "mode": "full_evaluation",
        "n_rollouts_failed": 0,
        "n_reproductions_failed": 1,
        "n_gradings_failed": 1,
    }

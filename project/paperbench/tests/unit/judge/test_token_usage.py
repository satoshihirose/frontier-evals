from paperbench.judge.graded_task_node import GradedTaskNode
from paperbench.judge.token_usage import get_total_token_usage


def test_total_token_usage_ignores_leaf_without_api_usage() -> None:
    leaf = GradedTaskNode(
        id="leaf",
        requirements="Implement the required method.",
        weight=1,
        sub_tasks=[],
        task_category="Code Development",
        score=1,
        valid_score=True,
        explanation="Satisfied.",
        judge_metadata={"token_usage": None},
    )

    assert get_total_token_usage(leaf).to_dict() == {}

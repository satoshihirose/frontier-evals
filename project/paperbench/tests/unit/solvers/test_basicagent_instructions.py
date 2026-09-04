from types import SimpleNamespace

import pytest

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

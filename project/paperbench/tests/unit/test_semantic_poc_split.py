from pathlib import Path
from typing import cast, get_args, get_type_hints

import pytest

from nanoeval.solvers.computer_tasks.code_execution_interface import ComputerInterface, NetworkMode
from paperbench.nano.eval import PaperBench
from paperbench.requirements import REQUIREMENTS_CONTAINER_PATH
from paperbench.solvers.codex.solver import CodexSolver
from paperbench.utils import get_experiments_dir


class RecordingComputer:
    def __init__(self) -> None:
        self.commands: list[str] = []
        self.uploads: dict[str, bytes] = {}

    async def check_shell_command(self, command: str) -> None:
        self.commands.append(command)

    async def upload(self, content: bytes, destination: str) -> None:
        self.uploads[destination] = content


def test_semantic_poc_split_contains_only_target_paper() -> None:
    split_path = get_experiments_dir() / "splits" / "semantic-poc.txt"

    assert split_path.read_text().splitlines() == ["semantic-self-consistency"]


def test_semantic_poc_is_an_accepted_paper_split() -> None:
    annotation = get_type_hints(PaperBench)["paper_split"]

    assert "semantic-poc" in get_args(annotation)
    assert Path(get_experiments_dir(), "splits", "semantic-poc.txt").exists()


@pytest.mark.parametrize(
    ("split_name", "paper_id"),
    [("bam-poc", "bam"), ("bbox-poc", "bbox")],
)
def test_additional_poc_splits_are_accepted(split_name: str, paper_id: str) -> None:
    annotation = get_type_hints(PaperBench)["paper_split"]
    split_path = Path(get_experiments_dir(), "splits", f"{split_name}.txt")

    assert split_name in get_args(annotation)
    assert split_path.read_text().splitlines() == [paper_id]


@pytest.mark.asyncio
async def test_semantic_poc_creates_one_networked_codex_task(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("paperbench.nano.eval.GRADER_OPENAI_API_KEY", "test-key")
    paperbench = PaperBench(
        paper_split="semantic-poc",
        solver=CodexSolver(),
        docker_image="pb-codex-env:latest",
        runs_dir=str(tmp_path),
    )

    tasks = await paperbench.get_instances()

    assert len(tasks) == 1
    assert tasks[0].paper_id == "semantic-self-consistency"
    assert tasks[0].docker_image == "pb-codex-env:latest"
    assert tasks[0].network_mode is NetworkMode.UNPROXIED
    assert tasks[0].requirements_csv is None
    prompt_content = tasks[0].prompt[0]["content"]
    assert isinstance(prompt_content, str)
    assert "additional list of reproduction requirements" not in prompt_content


@pytest.mark.asyncio
async def test_semantic_poc_adds_requirements_to_task_condition(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("paperbench.nano.eval.GRADER_OPENAI_API_KEY", "test-key")
    requirements_path = tmp_path / "requirements.csv"
    requirements_path.write_text(
        "requirement_id,requirement_statement,procedure_category\n"
        "req-001,The evaluation must use the full test set.,Experimental Inputs\n"
    )
    paperbench = PaperBench(
        paper_split="semantic-poc",
        solver=CodexSolver(),
        docker_image="pb-codex-env:latest",
        runs_dir=str(tmp_path / "runs"),
        requirements_csv=str(requirements_path),
    )

    tasks = await paperbench.get_instances()

    assert len(tasks) == 1
    assert tasks[0].requirements_csv == str(requirements_path.resolve())
    assert tasks[0].requirements_count == 1
    assert tasks[0].requirements_csv_sha256
    prompt_content = tasks[0].prompt[0]["content"]
    assert isinstance(prompt_content, str)
    assert REQUIREMENTS_CONTAINER_PATH in prompt_content

    computer = RecordingComputer()
    await tasks[0]._setup_requirements(cast(ComputerInterface, computer))

    assert computer.uploads == {
        REQUIREMENTS_CONTAINER_PATH: requirements_path.read_bytes(),
    }
    assert any(command.startswith("mkdir -p ") for command in computer.commands)
    assert f"chmod 0444 {REQUIREMENTS_CONTAINER_PATH}" in computer.commands
    assert tasks[0].requirements_input_metadata() == {
        "provided": True,
        "sha256": tasks[0].requirements_csv_sha256,
        "requirement_count": 1,
        "container_path": REQUIREMENTS_CONTAINER_PATH,
    }

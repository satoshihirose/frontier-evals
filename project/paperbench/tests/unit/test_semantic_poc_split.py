from pathlib import Path
from typing import cast

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


def test_paper_alias_registry_covers_every_paper() -> None:
    experiments_dir = get_experiments_dir()
    registry_path = experiments_dir / "paper-aliases.tsv"
    mappings = [
        line.split("\t")
        for line in registry_path.read_text().splitlines()
        if line and not line.startswith("#")
    ]
    aliases = [alias for alias, _ in mappings]
    identity_papers = {paper_id for alias, paper_id in mappings if alias == paper_id}
    paper_dirs = {
        path.name
        for path in (experiments_dir.parent / "data" / "papers").iterdir()
        if path.is_dir()
    }

    assert len(aliases) == len(set(aliases))
    assert identity_papers == paper_dirs
    assert [paper_id for alias, paper_id in mappings if alias == "semantic"] == [
        "semantic-self-consistency"
    ]
    assert list((experiments_dir / "splits").glob("*-poc.txt")) == []


@pytest.mark.asyncio
async def test_direct_paper_id_creates_one_networked_codex_task(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("paperbench.nano.eval.GRADER_OPENAI_API_KEY", "test-key")
    paperbench = PaperBench(
        paper_id="semantic-self-consistency",
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
async def test_direct_paper_id_creates_one_task_without_a_split(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("paperbench.nano.eval.GRADER_OPENAI_API_KEY", "test-key")
    paperbench = PaperBench(
        paper_id="semantic-self-consistency",
        paper_split="split-that-does-not-exist",
        solver=CodexSolver(),
        runs_dir=str(tmp_path),
    )

    tasks = await paperbench.get_instances()

    assert [task.paper_id for task in tasks] == ["semantic-self-consistency"]


@pytest.mark.asyncio
async def test_direct_paper_id_adds_requirements_to_task_condition(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("paperbench.nano.eval.GRADER_OPENAI_API_KEY", "test-key")
    requirements_path = tmp_path / "requirements.csv"
    requirements_path.write_text(
        "requirement_id,requirement_statement,procedure_category\n"
        "req-001,The evaluation must use the full test set.,Experimental Inputs\n"
    )
    paperbench = PaperBench(
        paper_id="semantic-self-consistency",
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

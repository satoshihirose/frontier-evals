from __future__ import annotations

import os
import subprocess
from pathlib import Path

from paperbench.utils import get_root


def test_roemia_launcher_externalizes_runs_cache_and_tmp(tmp_path: Path) -> None:
    launcher = get_root() / "scripts" / "run-bbox-roemia.sh"
    data_root = tmp_path / "paperbench-data"
    auth_file = tmp_path / "auth.json"
    agent_env = tmp_path / "agent.env"
    auth_file.write_text("{}\n")
    agent_env.write_text("OPENAI_API_KEY=test-placeholder\n")

    result = subprocess.run(
        [str(launcher), "--dry-run"],
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PAPERBENCH_DATA_ROOT": str(data_root),
            "CODEX_AUTH_FILE": str(auth_file),
            "PAPERBENCH_AGENT_ENV": str(agent_env),
        },
    )

    command = result.stdout
    assert f"paperbench.runs_dir={data_root / 'runs'}" in command
    assert (
        "paperbench.solver.computer_runtime.env.volumes_config.paperbench_cache."
        f"bind_source={data_root / 'cache'}"
    ) in command
    assert (
        "paperbench.solver.computer_runtime.env.volumes_config.paperbench_cache."
        "bind_dest=/root/.cache"
    ) in command
    assert (
        "paperbench.reproduction.computer_runtime.env.volumes_config.paperbench_cache."
        f"bind_source={data_root / 'cache'}"
    ) in command
    assert (
        "paperbench.reproduction.computer_runtime.env.volumes_config.paperbench_tmp."
        f"bind_source={data_root / 'tmp' / 'reproduction'}"
    ) in command
    assert "paperbench.solver.computer_runtime.env.is_nvidia_gpu_env=true" in command
    assert "paperbench.reproduction.computer_runtime.env.is_nvidia_gpu_env=true" in command
    assert "paperbench.paper_split=bbox-poc" in command
    assert "paperbench.judge.code_only=False" in command
    assert "paperbench.reproduction.skip_reproduction=False" in command
    assert "test-placeholder" not in command


def test_roemia_launcher_adds_requirements_only_when_requested(tmp_path: Path) -> None:
    launcher = get_root() / "scripts" / "run-bbox-roemia.sh"
    auth_file = tmp_path / "auth.json"
    agent_env = tmp_path / "agent.env"
    requirements = tmp_path / "requirements.csv"
    auth_file.write_text("{}\n")
    agent_env.write_text("OPENAI_API_KEY=test-placeholder\n")
    requirements.write_text(
        "requirement_id,requirement_statement,procedure_category\n"
        "req-001,The test must run.,Method Configuration\n"
    )
    environment = {
        **os.environ,
        "PAPERBENCH_DATA_ROOT": str(tmp_path / "data"),
        "CODEX_AUTH_FILE": str(auth_file),
        "PAPERBENCH_AGENT_ENV": str(agent_env),
    }

    baseline = subprocess.run(
        [str(launcher), "--dry-run"],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    with_requirements = subprocess.run(
        [str(launcher), "--dry-run", "--requirements-csv", str(requirements)],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert "paperbench.requirements_csv=" not in baseline.stdout
    assert f"paperbench.requirements_csv={requirements.resolve()}" in with_requirements.stdout

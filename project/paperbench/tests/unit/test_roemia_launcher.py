from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from paperbench.utils import get_root


def test_paper_asset_materializer_fetches_only_the_requested_lfs_directory(
    tmp_path: Path,
) -> None:
    helper = get_root() / "scripts" / "materialize-paper-assets.sh"
    repo_root = tmp_path / "frontier-evals"
    paperbench_root = repo_root / "project" / "paperbench"
    paper_dir = paperbench_root / "data" / "papers" / "example-paper"
    fake_bin = tmp_path / "bin"
    lfs_args = tmp_path / "git-lfs-args.txt"
    paper_dir.mkdir(parents=True)
    fake_bin.mkdir()
    subprocess.run(["git", "init", "-q", str(repo_root)], check=True)
    (paper_dir / "paper.md").write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:0000000000000000000000000000000000000000000000000000000000000000\n"
        "size 100\n"
    )
    fake_git_lfs = fake_bin / "git-lfs"
    fake_git_lfs.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$@\" > \"$GIT_LFS_ARGS\"\n"
        "printf '%s\\n' title one two three four five six > "
        "project/paperbench/data/papers/example-paper/paper.md\n"
    )
    fake_git_lfs.chmod(0o755)

    subprocess.run(
        [
            str(helper),
            "--paper",
            "example-paper",
            "--paperbench-root",
            str(paperbench_root),
        ],
        check=True,
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "GIT_LFS_ARGS": str(lfs_args),
        },
    )

    assert (paper_dir / "paper.md").read_text().splitlines() == [
        "title",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
    ]
    assert lfs_args.read_text().splitlines() == [
        "pull",
        "--include=project/paperbench/data/papers/example-paper/**",
    ]


def test_paper_asset_materializer_skips_an_already_materialized_paper(
    tmp_path: Path,
) -> None:
    helper = get_root() / "scripts" / "materialize-paper-assets.sh"
    paperbench_root = tmp_path / "frontier-evals" / "project" / "paperbench"
    paper_dir = paperbench_root / "data" / "papers" / "example-paper"
    fake_bin = tmp_path / "bin"
    paper_dir.mkdir(parents=True)
    fake_bin.mkdir()
    (paper_dir / "paper.md").write_text("title\none\ntwo\nthree\nfour\nfive\n")
    fake_git_lfs = fake_bin / "git-lfs"
    fake_git_lfs.write_text("#!/usr/bin/env bash\nexit 99\n")
    fake_git_lfs.chmod(0o755)

    subprocess.run(
        [
            str(helper),
            "--paper",
            "example-paper",
            "--paperbench-root",
            str(paperbench_root),
        ],
        check=True,
        env={**os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}"},
    )


def test_roemia_launcher_requires_explicit_standalone_mode() -> None:
    launcher = get_root() / "scripts" / "run-paperbench-roemia.sh"

    result = subprocess.run(
        [str(launcher), "--paper", "bam"],
        check=False,
        capture_output=True,
        text=True,
        env=os.environ,
    )

    assert result.returncode == 2
    assert "Use run_paperbench_with_qwen_judge.sh by default" in result.stderr
    assert "--standalone" in result.stderr


def test_roemia_launcher_discovers_new_paper_alias_from_registry(
    tmp_path: Path,
) -> None:
    source_launcher = get_root() / "scripts" / "run-paperbench-roemia.sh"
    paperbench_root = tmp_path / "paperbench"
    scripts_dir = paperbench_root / "paperbench" / "scripts"
    experiments_dir = paperbench_root / "experiments"
    scripts_dir.mkdir(parents=True)
    experiments_dir.mkdir(parents=True)
    launcher = scripts_dir / source_launcher.name
    shutil.copy2(source_launcher, launcher)
    materializer = scripts_dir / "materialize-paper-assets.sh"
    materializer.write_text("#!/usr/bin/env bash\nexit 0\n")
    materializer.chmod(0o755)
    (experiments_dir / "paper-aliases.tsv").write_text("new\tnew-paper\n")
    auth_file = tmp_path / "auth.json"
    agent_env = tmp_path / "agent.env"
    auth_file.write_text("{}\n")
    agent_env.write_text("OPENAI_API_KEY=test-placeholder\n")

    result = subprocess.run(
        [str(launcher), "--dry-run", "--paper", "new"],
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PAPERBENCH_DATA_ROOT": str(tmp_path / "data"),
            "CODEX_AUTH_FILE": str(auth_file),
            "PAPERBENCH_AGENT_ENV": str(agent_env),
        },
    )

    assert "paperbench.paper_id=new-paper" in result.stdout
    assert "paperbench.paper_split=" not in result.stdout


def test_roemia_launcher_externalizes_runs_cache_and_tmp(tmp_path: Path) -> None:
    launcher = get_root() / "scripts" / "run-paperbench-roemia.sh"
    data_root = tmp_path / "paperbench-data"
    auth_file = tmp_path / "auth.json"
    agent_env = tmp_path / "agent.env"
    auth_file.write_text("{}\n")
    agent_env.write_text("OPENAI_API_KEY=test-placeholder\n")

    result = subprocess.run(
        [str(launcher), "--dry-run", "--paper", "bam"],
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
    assert (
        command.count("paperbench.solver.computer_runtime.env.gpu_device_id=") == 1
    )
    assert (
        command.count("paperbench.reproduction.computer_runtime.env.gpu_device_id=")
        == 1
    )
    assert "paperbench.paper_id=bam" in command
    assert "paperbench.judge.code_only=False" in command
    assert "paperbench.reproduction.skip_reproduction=False" in command
    assert "paperbench.solver.persist_refreshed_auth=true" in command
    assert (
        f"paperbench.solver.codex_auth_file={data_root / 'tmp' / 'codex-auth' / 'dry-run' / 'auth.json'}"
        in command
    )
    assert (
        f"paperbench.judge.completer_config.codex_home={data_root / 'tmp' / 'codex-auth' / 'dry-run'}"
        in command
    )
    assert "test-placeholder" not in command


def test_roemia_launcher_uses_configured_gpu_server_data_base(
    tmp_path: Path,
) -> None:
    launcher = get_root() / "scripts" / "run-paperbench-roemia.sh"
    data_base = tmp_path / "mnt-data1"
    expected_root = data_base / "test-user" / "paperbench"
    auth_file = tmp_path / "auth.json"
    agent_env = tmp_path / "agent.env"
    auth_file.write_text("{}\n")
    agent_env.write_text("OPENAI_API_KEY=test-placeholder\n")

    result = subprocess.run(
        [str(launcher), "--dry-run", "--paper", "bam"],
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "USER": "test-user",
            "PAPERBENCH_DATA_BASE": str(data_base),
            "CODEX_AUTH_FILE": str(auth_file),
            "PAPERBENCH_AGENT_ENV": str(agent_env),
        },
    )

    assert f"paperbench.runs_dir={expected_root / 'runs'}" in result.stdout


def test_roemia_launcher_dry_run_does_not_require_credentials(
    tmp_path: Path,
) -> None:
    launcher = get_root() / "scripts" / "run-paperbench-roemia.sh"

    result = subprocess.run(
        [str(launcher), "--dry-run", "--paper", "bam"],
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PAPERBENCH_DATA_ROOT": str(tmp_path / "data"),
            "CODEX_AUTH_FILE": str(tmp_path / "missing-auth.json"),
            "PAPERBENCH_AGENT_ENV": str(tmp_path / "missing-agent.env"),
        },
    )

    assert "paperbench.paper_id=bam" in result.stdout


def test_roemia_launcher_adds_requirements_only_when_requested(tmp_path: Path) -> None:
    launcher = get_root() / "scripts" / "run-paperbench-roemia.sh"
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
        [str(launcher), "--dry-run", "--paper", "bam"],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    with_requirements = subprocess.run(
        [
            str(launcher),
            "--dry-run",
            "--paper",
            "bam",
            "--requirements-csv",
            str(requirements),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert "paperbench.requirements_csv=" not in baseline.stdout
    assert f"paperbench.requirements_csv={requirements.resolve()}" in with_requirements.stdout


def test_roemia_launcher_selects_one_idle_gpu_by_uuid(tmp_path: Path) -> None:
    launcher = get_root() / "scripts" / "run-paperbench-roemia.sh"
    auth_file = tmp_path / "auth.json"
    agent_env = tmp_path / "agent.env"
    data_root = tmp_path / "data"
    bin_dir = tmp_path / "bin"
    auth_file.write_text("{}\n")
    agent_env.write_text("OPENAI_API_KEY=test-placeholder\n")
    bin_dir.mkdir()

    nvidia_smi = bin_dir / "nvidia-smi"
    nvidia_smi.write_text(
        """#!/usr/bin/env bash
case "$*" in
  *--query-compute-apps=gpu_uuid*)
    printf '%s\\n' GPU-busy
    ;;
  *--query-gpu=index,uuid,memory.free*)
    printf '%s\\n' \
      '0, GPU-busy, 70000' \
      '1, GPU-idle-smaller, 60000' \
      '2, GPU-idle-largest, 80000'
    ;;
  *)
    exit 2
    ;;
esac
"""
    )
    nvidia_smi.chmod(0o755)
    result = subprocess.run(
        [str(launcher), "--dry-run", "--paper", "bam"],
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "PAPERBENCH_DATA_ROOT": str(data_root),
            "CODEX_AUTH_FILE": str(auth_file),
            "PAPERBENCH_AGENT_ENV": str(agent_env),
        },
    )

    assert "Selected GPU 2 (GPU-idle-largest" in result.stderr
    assert (
        result.stdout.count(
            "computer_runtime.env.gpu_device_id=GPU-idle-largest"
        )
        == 2
    )


def test_roemia_launcher_rejects_unknown_paper(tmp_path: Path) -> None:
    launcher = get_root() / "scripts" / "run-paperbench-roemia.sh"
    auth_file = tmp_path / "auth.json"
    agent_env = tmp_path / "agent.env"
    auth_file.write_text("{}\n")
    agent_env.write_text("OPENAI_API_KEY=test-placeholder\n")

    result = subprocess.run(
        [str(launcher), "--dry-run", "--paper", "unknown"],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PAPERBENCH_DATA_ROOT": str(tmp_path / "data"),
            "CODEX_AUTH_FILE": str(auth_file),
            "PAPERBENCH_AGENT_ENV": str(agent_env),
        },
    )

    assert result.returncode == 2
    assert "Unsupported paper alias: unknown" in result.stderr


@pytest.mark.parametrize(
    ("paper_alias", "paper_id"),
    [
        ("adaptive-pruning", "adaptive-pruning"),
        ("bam", "bam"),
        ("bbox", "bbox"),
        ("mechanistic-understanding", "mechanistic-understanding"),
        ("semantic", "semantic-self-consistency"),
        ("semantic-self-consistency", "semantic-self-consistency"),
        ("stochastic-interpolants", "stochastic-interpolants"),
    ],
)
def test_roemia_launcher_resolves_paper_alias_to_direct_paper_id(
    tmp_path: Path,
    paper_alias: str,
    paper_id: str,
) -> None:
    launcher = get_root() / "scripts" / "run-paperbench-roemia.sh"
    auth_file = tmp_path / "auth.json"
    agent_env = tmp_path / "agent.env"
    auth_file.write_text("{}\n")
    agent_env.write_text("OPENAI_API_KEY=test-placeholder\n")

    result = subprocess.run(
        [str(launcher), "--dry-run", "--paper", paper_alias],
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PAPERBENCH_DATA_ROOT": str(tmp_path / "data"),
            "CODEX_AUTH_FILE": str(auth_file),
            "PAPERBENCH_AGENT_ENV": str(agent_env),
        },
    )

    assert f"paperbench.paper_id={paper_id}" in result.stdout


def test_alcatraz_limits_nvidia_container_to_configured_gpu() -> None:
    from alcatraz.clusters.local import LocalConfig

    gpu_uuid = "GPU-12345678"
    cluster = LocalConfig(
        is_nvidia_gpu_env=True,
        gpu_device_id=gpu_uuid,
    ).build()

    requests = cluster._gpu_device_requests()

    assert len(requests) == 1
    assert requests[0]["DeviceIDs"] == [gpu_uuid]
    assert requests[0]["Count"] == 0

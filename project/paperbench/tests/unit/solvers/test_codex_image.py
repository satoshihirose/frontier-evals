from __future__ import annotations

from pathlib import Path

PAPERBENCH_ROOT = Path(__file__).parents[3]


def test_codex_image_pins_cli_and_extends_paperbench_agent_image() -> None:
    dockerfile = (
        PAPERBENCH_ROOT / "paperbench" / "solvers" / "codex" / "Dockerfile.agent"
    ).read_text()

    assert "ARG CODEX_VERSION=0.144.2" in dockerfile
    assert 'npm install --global "@openai/codex@${CODEX_VERSION}"' in dockerfile
    assert "FROM pb-env:latest" in dockerfile
    assert "RUN mkdir -p /home/.codex" in dockerfile
    assert "RUN codex --version" in dockerfile


def test_build_script_builds_base_before_codex_image() -> None:
    script = (PAPERBENCH_ROOT / "paperbench" / "scripts" / "build-docker-images.sh").read_text()

    base_build = script.index("-t pb-env")
    codex_build = script.index("-t pb-codex-env")
    reproducer_build = script.index("-t pb-reproducer")

    assert base_build < codex_build < reproducer_build


def test_reproducer_exposes_python_pip_and_jupyter_commands_for_alcatraz() -> None:
    dockerfile = (PAPERBENCH_ROOT / "paperbench" / "reproducer.Dockerfile").read_text()

    assert "python3-pip" in dockerfile
    assert "jupyter" in dockerfile
    assert "/usr/local/bin/python" in dockerfile
    assert "/usr/local/bin/pip" in dockerfile
    assert "RUN jupyter --version" in dockerfile

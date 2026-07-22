from pathlib import Path

from paperbench.judge.simple import normalize_selected_file_paths


def test_file_selection_normalizes_cli_absolute_paths_to_submission_paths() -> None:
    available = [
        Path("src/semantic_sc/data.py"),
        Path("configs/paper.json"),
        Path("README.md"),
    ]
    selected = "\n".join(
        [
            "/tmp/paperbench-codex-judge/src/semantic_sc/data.py",
            "/tmp/paperbench-codex-judge/configs/paper.json",
            "README.md",
            "/etc/passwd",
        ]
    )

    assert normalize_selected_file_paths(selected, available, max_files=10) == [
        Path("src/semantic_sc/data.py"),
        Path("configs/paper.json"),
        Path("README.md"),
    ]

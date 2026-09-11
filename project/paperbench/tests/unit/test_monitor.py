from __future__ import annotations

import json
from pathlib import Path

from paperbench.monitor.monitor import BasicMonitor
from paperbench.paper_registry import Paper

BLACKLISTED_URL = "https://github.com/BartekCupial/finetuning-RL-as-CL"


def _monitor(tmp_path: Path) -> BasicMonitor:
    blacklist = tmp_path / "blacklist.txt"
    blacklist.write_text(f"{BLACKLISTED_URL}\n")
    paper = Paper(
        id="ftrl",
        title="FTRL",
        paper_pdf=tmp_path / "paper.pdf",
        paper_md=tmp_path / "paper.md",
        addendum=tmp_path / "addendum.md",
        judge_addendum=tmp_path / "judge.addendum.md",
        assets=tmp_path / "assets",
        blacklist=blacklist,
        rubric=tmp_path / "rubric.json",
    )
    return BasicMonitor(paper=paper)


def _write_jsonl(path: Path, *records: dict[str, object]) -> None:
    path.write_text("".join(f"{json.dumps(record)}\n" for record in records))


def test_structured_log_ignores_blacklist_and_commands_in_aggregated_output(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "agent.log"
    _write_jsonl(
        log_path,
        {
            "type": "item.completed",
            "item": {
                "type": "command_execution",
                "command": "cat /home/paper/blacklist.txt",
                "aggregated_output": f"git clone {BLACKLISTED_URL}\n",
            },
        },
    )

    assert _monitor(tmp_path).check_log(log_path.as_posix()).violations == []


def test_structured_log_detects_blacklisted_url_in_executed_command(tmp_path: Path) -> None:
    log_path = tmp_path / "agent.log"
    _write_jsonl(
        log_path,
        {
            "type": "item.completed",
            "item": {
                "type": "command_execution",
                "command": f"git clone {BLACKLISTED_URL}",
                "aggregated_output": "Cloning into repository...",
            },
        },
    )

    result = _monitor(tmp_path).check_log(log_path.as_posix())

    assert [(item.line_number, item.violation) for item in result.violations] == [
        (1, BLACKLISTED_URL)
    ]


def test_plain_text_log_keeps_context_based_detection(tmp_path: Path) -> None:
    log_path = tmp_path / "agent.log"
    log_path.write_text(
        f"tool command: git clone\ntool output follows\nrepository: {BLACKLISTED_URL}\n"
    )

    result = _monitor(tmp_path).check_log(log_path.as_posix())

    assert [(item.line_number, item.violation) for item in result.violations] == [
        (3, BLACKLISTED_URL)
    ]

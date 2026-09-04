import hashlib
import json
from pathlib import Path

import pytest

from paperbench.evaluation_specification import load_rubric_criteria_artifact
from paperbench.paper_registry import paper_registry
from paperbench.scripts.generate_rubric_criteria import (
    MANIFEST_FILENAME,
    generate_all_rubric_criteria,
    verify_all_rubric_criteria,
)


def _write_paper(papers_dir: Path, paper_id: str, requirement: str) -> None:
    paper_dir = papers_dir / paper_id
    paper_dir.mkdir(parents=True)
    (paper_dir / "config.yaml").write_text(f'id: {paper_id}\ntitle: "{paper_id}"\n')
    (paper_dir / "rubric.json").write_text(
        json.dumps(
            {
                "sub_tasks": [
                    {"requirements": requirement, "sub_tasks": []},
                ]
            }
        )
    )


def test_generate_and_verify_all_registered_papers(tmp_path: Path) -> None:
    papers_dir = tmp_path / "papers"
    artifact_dir = tmp_path / "rubric-criteria"
    _write_paper(papers_dir, "paper-a", "Criterion A")
    _write_paper(papers_dir, "paper-b", "Criterion B")

    result = generate_all_rubric_criteria(papers_dir, artifact_dir)

    assert result.paper_count == 2
    assert json.loads((artifact_dir / "paper-a.json").read_text()) == [
        "Criterion A"
    ]
    manifest = json.loads((artifact_dir / MANIFEST_FILENAME).read_text())
    assert sorted(manifest["papers"]) == ["paper-a", "paper-b"]
    assert manifest["papers"]["paper-a"]["criterion_count"] == 1
    verify_all_rubric_criteria(papers_dir, artifact_dir)


def test_invalid_source_does_not_partially_replace_existing_artifacts(tmp_path: Path) -> None:
    papers_dir = tmp_path / "papers"
    artifact_dir = tmp_path / "rubric-criteria"
    _write_paper(papers_dir, "paper-a", "Old A")
    _write_paper(papers_dir, "paper-b", "Old B")
    generate_all_rubric_criteria(papers_dir, artifact_dir)
    old_artifact = (artifact_dir / "paper-a.json").read_bytes()
    old_manifest = (artifact_dir / MANIFEST_FILENAME).read_bytes()

    (papers_dir / "paper-a" / "rubric.json").write_text(
        json.dumps({"sub_tasks": [{"requirements": "New A", "sub_tasks": []}]})
    )
    (papers_dir / "paper-b" / "rubric.json").write_text(
        json.dumps({"sub_tasks": [{"sub_tasks": []}]})
    )

    with pytest.raises(ValueError, match="paper-b"):
        generate_all_rubric_criteria(papers_dir, artifact_dir)

    assert (artifact_dir / "paper-a.json").read_bytes() == old_artifact
    assert (artifact_dir / MANIFEST_FILENAME).read_bytes() == old_manifest


def test_stale_artifact_requires_full_regeneration(tmp_path: Path) -> None:
    papers_dir = tmp_path / "papers"
    artifact_dir = tmp_path / "rubric-criteria"
    _write_paper(papers_dir, "paper-a", "Old criterion")
    generate_all_rubric_criteria(papers_dir, artifact_dir)
    (papers_dir / "paper-a" / "rubric.json").write_text(
        json.dumps({"sub_tasks": [{"requirements": "New criterion", "sub_tasks": []}]})
    )

    with pytest.raises(ValueError, match="regenerate all rubric criteria"):
        verify_all_rubric_criteria(papers_dir, artifact_dir)

    generate_all_rubric_criteria(papers_dir, artifact_dir)
    verify_all_rubric_criteria(papers_dir, artifact_dir)
    assert json.loads((artifact_dir / "paper-a.json").read_text()) == [
        "New criterion"
    ]


def test_runtime_loader_rejects_artifact_not_recorded_by_manifest(tmp_path: Path) -> None:
    papers_dir = tmp_path / "papers"
    artifact_dir = tmp_path / "rubric-criteria"
    _write_paper(papers_dir, "paper-a", "Criterion A")
    generate_all_rubric_criteria(papers_dir, artifact_dir)
    artifact = artifact_dir / "paper-a.json"
    artifact.write_text(json.dumps(["Changed after generation"]))

    with pytest.raises(ValueError, match="regenerate all rubric criteria"):
        load_rubric_criteria_artifact(
            paper_id="paper-a",
            rubric_path=papers_dir / "paper-a" / "rubric.json",
        )


def test_manifest_records_source_and_artifact_hashes(tmp_path: Path) -> None:
    papers_dir = tmp_path / "papers"
    artifact_dir = tmp_path / "rubric-criteria"
    _write_paper(papers_dir, "paper-a", "Criterion A")
    generate_all_rubric_criteria(papers_dir, artifact_dir)

    manifest = json.loads((artifact_dir / MANIFEST_FILENAME).read_text())
    entry = manifest["papers"]["paper-a"]
    rubric = papers_dir / "paper-a" / "rubric.json"
    artifact = artifact_dir / "paper-a.json"
    assert entry["rubric_sha256"] == hashlib.sha256(rubric.read_bytes()).hexdigest()
    assert entry["rubric_criteria_sha256"] == hashlib.sha256(artifact.read_bytes()).hexdigest()


def test_checked_in_artifacts_cover_every_registered_paper() -> None:
    result = verify_all_rubric_criteria(paper_registry.get_papers_dir())

    assert result.paper_count == len(paper_registry.list_paper_ids()) == 23

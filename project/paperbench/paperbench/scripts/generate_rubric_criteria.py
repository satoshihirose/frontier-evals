from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from paperbench.evaluation_specification import (
    RUBRIC_CRITERIA_ARTIFACT_DIRNAME,
    RUBRIC_CRITERIA_MANIFEST_FILENAME,
    flatten_rubric_criteria,
)
from paperbench.paper_registry import paper_registry

MANIFEST_FILENAME = RUBRIC_CRITERIA_MANIFEST_FILENAME


@dataclass(frozen=True)
class GenerationResult:
    paper_count: int
    criterion_count: int


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _registered_paper_directories(papers_dir: Path) -> list[Path]:
    return sorted(config.parent for config in papers_dir.rglob("config.yaml"))


def _build_all(
    papers_dir: Path, artifact_dir: Path
) -> tuple[dict[Path, bytes], bytes, GenerationResult]:
    artifacts: dict[Path, bytes] = {}
    manifest_entries: dict[str, dict[str, str | int]] = {}
    total_criteria = 0
    paper_directories = _registered_paper_directories(papers_dir)
    if not paper_directories:
        raise ValueError(f"No registered papers found under {papers_dir}")

    for paper_dir in paper_directories:
        paper_id = paper_dir.name
        rubric_path = paper_dir / "rubric.json"
        try:
            rubric_content = rubric_path.read_bytes()
            rubric = json.loads(rubric_content)
            if not isinstance(rubric, dict):
                raise ValueError("rubric root must be an object")
            criteria = flatten_rubric_criteria(rubric)
        except (FileNotFoundError, json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"Cannot generate rubric criteria for {paper_id}: {error}") from error

        artifact = (json.dumps(criteria, ensure_ascii=False, indent=2) + "\n").encode()
        artifacts[artifact_dir / f"{paper_id}.json"] = artifact
        manifest_entries[paper_id] = {
            "rubric_sha256": _sha256(rubric_content),
            "rubric_criteria_sha256": _sha256(artifact),
            "criterion_count": len(criteria),
        }
        total_criteria += len(criteria)

    manifest = {
        "schema_version": 1,
        "papers": manifest_entries,
    }
    manifest_content = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode()
    return (
        artifacts,
        manifest_content,
        GenerationResult(len(paper_directories), total_criteria),
    )


def generate_all_rubric_criteria(
    papers_dir: Path, artifact_dir: Path | None = None
) -> GenerationResult:
    """Regenerate every registered paper after validating every source rubric."""
    resolved_artifact_dir = artifact_dir or papers_dir.parent / RUBRIC_CRITERIA_ARTIFACT_DIRNAME
    artifacts, manifest, result = _build_all(papers_dir, resolved_artifact_dir)
    temporary_paths: list[Path] = []
    try:
        resolved_artifact_dir.mkdir(parents=True, exist_ok=True)
        for destination, content in artifacts.items():
            temporary = destination.with_name(f".{destination.name}.tmp")
            temporary.write_bytes(content)
            temporary_paths.append(temporary)
        manifest_path = resolved_artifact_dir / MANIFEST_FILENAME
        temporary_manifest = manifest_path.with_name(f".{manifest_path.name}.tmp")
        temporary_manifest.write_bytes(manifest)
        temporary_paths.append(temporary_manifest)

        for destination in artifacts:
            os.replace(destination.with_name(f".{destination.name}.tmp"), destination)
        os.replace(temporary_manifest, manifest_path)
    finally:
        for temporary in temporary_paths:
            temporary.unlink(missing_ok=True)
    return result


def verify_all_rubric_criteria(
    papers_dir: Path, artifact_dir: Path | None = None
) -> GenerationResult:
    resolved_artifact_dir = artifact_dir or papers_dir.parent / RUBRIC_CRITERIA_ARTIFACT_DIRNAME
    expected_artifacts, expected_manifest, result = _build_all(papers_dir, resolved_artifact_dir)
    manifest_path = resolved_artifact_dir / MANIFEST_FILENAME
    if not manifest_path.is_file() or manifest_path.read_bytes() != expected_manifest:
        raise ValueError(
            "Rubric criteria manifest is missing or stale; regenerate all rubric criteria with "
            "`python -m paperbench.scripts.generate_rubric_criteria`"
        )
    for path, expected in expected_artifacts.items():
        if not path.is_file() or path.read_bytes() != expected:
            raise ValueError(
                f"Rubric criteria artifact is missing or stale for {path.parent.name}; "
                "regenerate all rubric criteria with "
                "`python -m paperbench.scripts.generate_rubric_criteria`"
            )
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Generate or verify criteria-only artifacts for every registered paper."
    )
    result.add_argument(
        "--papers-dir",
        type=Path,
        default=paper_registry.get_papers_dir(),
        help="Paper registry directory (defaults to PaperBench data/papers).",
    )
    result.add_argument(
        "--check",
        action="store_true",
        help="Verify all artifacts without modifying them.",
    )
    result.add_argument(
        "--output-dir",
        type=Path,
        help="Generated artifact directory (defaults to data/rubric-criteria).",
    )
    return result


def main() -> None:
    args = parser().parse_args()
    operation = verify_all_rubric_criteria if args.check else generate_all_rubric_criteria
    result = operation(
        args.papers_dir.resolve(),
        args.output_dir.resolve() if args.output_dir else None,
    )
    verb = "Verified" if args.check else "Generated"
    print(f"{verb} {result.paper_count} papers and {result.criterion_count} criteria.")


if __name__ == "__main__":
    main()

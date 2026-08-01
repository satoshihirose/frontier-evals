from __future__ import annotations

import asyncio
import hashlib
import json
from typing import Any

import blobfile as bf
import structlog.stdlib

from paperbench.judge.graded_task_node import GradedTaskNode
from paperbench.rubric.tasks import TaskNode

logger = structlog.stdlib.get_logger(component=__name__)

CHECKPOINT_FORMAT_VERSION = 1


def stable_hash(value: Any) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


class LeafCheckpointStore:
    """Append-only storage for successfully graded rubric leaves."""

    def __init__(self, *, path: str, context_hash: str) -> None:
        self.path = path
        self.context_hash = context_hash
        self._records: dict[str, dict[str, Any]] = {}
        self._loaded = False
        self._lock = asyncio.Lock()

    async def load(self) -> None:
        async with self._lock:
            if self._loaded:
                return
            if bf.exists(self.path):
                with bf.BlobFile(self.path, "r") as checkpoint_file:
                    for line_number, line in enumerate(checkpoint_file, start=1):
                        if not line.strip():
                            continue
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            logger.warning(
                                "Ignoring incomplete leaf checkpoint record",
                                path=self.path,
                                line_number=line_number,
                            )
                            continue
                        if (
                            record.get("format_version") == CHECKPOINT_FORMAT_VERSION
                            and record.get("context_hash") == self.context_hash
                            and isinstance(record.get("task_id"), str)
                        ):
                            self._records[record["task_id"]] = record
            self._loaded = True

    def get(self, task: TaskNode) -> GradedTaskNode | None:
        if not self._loaded:
            raise RuntimeError("Leaf checkpoint store must be loaded before use")
        record = self._records.get(task.id)
        if record is None or record.get("task_hash") != stable_hash(task.to_dict()):
            return None
        try:
            graded = GradedTaskNode.from_dict(record["graded_task"])
        except (KeyError, TypeError, ValueError):
            logger.warning(
                "Ignoring invalid leaf checkpoint record",
                path=self.path,
                task_id=task.id,
            )
            return None
        if not graded.is_leaf() or not graded.valid_score:
            return None
        return graded

    async def save(self, task: TaskNode, graded: GradedTaskNode) -> None:
        if not graded.valid_score:
            return
        record = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "context_hash": self.context_hash,
            "task_id": task.id,
            "task_hash": stable_hash(task.to_dict()),
            "graded_task": graded.to_dict(),
        }
        serialized = "\n" + json.dumps(record, ensure_ascii=False, sort_keys=True)
        async with self._lock:
            with bf.BlobFile(self.path, "a") as checkpoint_file:
                checkpoint_file.write(serialized)
            self._records[task.id] = record

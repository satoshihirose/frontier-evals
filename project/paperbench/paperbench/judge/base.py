import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

import openai
import structlog.stdlib
from structlog import wrap_logger
from structlog.stdlib import BoundLogger

from nanoeval.solvers.computer_tasks.code_execution_interface import ComputerInterface
from paperbench.judge.graded_task_node import GradedTaskNode, score_from_children
from paperbench.judge.leaf_checkpoint import LeafCheckpointStore
from paperbench.judge.utils import file_exists, read_file_content, read_file_mtime, reduce_log
from paperbench.rubric.tasks import TaskNode

logger = structlog.stdlib.get_logger(component=__name__)


class Judge(ABC):
    def __init__(
        self,
        paper_path: Path,
        rubric: TaskNode,
        addendum: str | None,
        judge_addendum: str | None,
        submission_dir: Path,
        log_path: Path | None = None,
        max_depth: int = 999,
        code_only: bool = False,
        computer: ComputerInterface | None = None,
        leaf_checkpoint_store: LeafCheckpointStore | None = None,
        max_invalid_leaf_retries: int = 3,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.paper_path: Path = paper_path
        self.rubric: TaskNode = rubric
        self.addendum: str | None = addendum
        self.judge_addendum: str | None = judge_addendum
        self.submission_dir: Path = submission_dir
        self.log_path: Path | None = log_path
        self.max_depth: int = max_depth
        self.code_only: bool = code_only
        self.computer = computer
        self.leaf_checkpoint_store = leaf_checkpoint_store
        if max_invalid_leaf_retries < 0:
            raise ValueError("max_invalid_leaf_retries must be non-negative")
        self.max_invalid_leaf_retries = max_invalid_leaf_retries

        # Reproduction script and log
        self.reproduce_sh_path: Path = self.submission_dir / "reproduce.sh"
        self.reproduce_log_path: Path = self.submission_dir / "reproduce.log"
        self.reproduce_log_creation_file_path: Path = (
            self.submission_dir / "reproduce.log.creation_time"
        )
        # initialize with fallback values, we'll try populating in populate_file_content
        self.reproduce_sh_content = "(Does not exist)"
        self.reproduce_log_content = "(Does not exist)"
        self.reproduction_log_creation_time_utc = datetime.now(tz=timezone.utc)

    @property
    @abstractmethod
    def judge_type(self) -> str:
        """Abstract property for judge type, to be implemented by sub-classes."""
        raise NotImplementedError()

    async def before_grading(self) -> None:
        """
        Hook for any setup before grading starts.
        Separated from __init__ to allow for async operations.
        """
        await self.read_repro_files_content()
        if self.leaf_checkpoint_store is not None:
            await self.leaf_checkpoint_store.load()

    async def read_repro_files_content(self) -> None:
        """
        Asynchronously reads reproduce.sh, reproduce.log, and reproduce.log.creation_time
        """
        reproduce_sh_exists = await file_exists(self.reproduce_sh_path, self.computer)
        reproduce_log_exists = await file_exists(self.reproduce_log_path, self.computer)
        reproduce_log_creation_file_exists = await file_exists(
            self.reproduce_log_creation_file_path, self.computer
        )

        if reproduce_sh_exists:
            self.reproduce_sh_content = await read_file_content(
                self.reproduce_sh_path, self.computer
            )

        if reproduce_log_exists:
            self.reproduce_log_content = await read_file_content(
                self.reproduce_log_path, self.computer
            )
            self.reproduce_log_content = reduce_log(self.reproduce_log_content)
            # improve the creation time utc fallback
            self.reproduction_log_creation_time_utc = datetime.fromtimestamp(
                await read_file_mtime(self.reproduce_log_path, self.computer), tz=timezone.utc
            )

        if reproduce_log_creation_file_exists:
            timestamp = await read_file_content(
                self.reproduce_log_creation_file_path, self.computer
            )
            self.reproduction_log_creation_time_utc = datetime.fromtimestamp(
                int(timestamp), tz=timezone.utc
            )

    async def judge(
        self,
        root_task: TaskNode | None = None,
        grade_leaf_fn: Callable[[TaskNode], Awaitable[GradedTaskNode]] | None = None,
    ) -> GradedTaskNode:
        """
        Grades an entire task tree by calling self.grade recursively.
        Ensures self.before_grading() is called once before grading starts.
        """
        await self.before_grading()

        grade_leaf_fn = grade_leaf_fn or self.grade_leaf
        if self.leaf_checkpoint_store is not None:
            uncached_grade_leaf_fn = grade_leaf_fn

            async def grade_leaf_with_checkpoint(task: TaskNode) -> GradedTaskNode:
                assert self.leaf_checkpoint_store is not None
                checkpoint = self.leaf_checkpoint_store.get(task)
                if checkpoint is not None:
                    logger.info(f"Reusing completed grade for leaf {task.id}")
                    return checkpoint
                graded = await uncached_grade_leaf_fn(task)
                await self.leaf_checkpoint_store.save(task, graded)
                return graded

            grade_leaf_fn = grade_leaf_with_checkpoint

        if root_task is None:
            root_task = self.rubric

        return await self.grade(root_task, grade_leaf_fn)

    async def grade(
        self,
        task: TaskNode,
        grade_leaf_fn: Callable[[TaskNode], Awaitable[GradedTaskNode]],
        current_depth: int = 1,
    ) -> GradedTaskNode:
        """
        Options:
        - task: The (sub)task to grade. If None, the entire rubric is graded.
        - grade_leaf_fn: The function to use to grade leaf nodes. If None, the default `grade_leaf` method is used.

        Returns a `GradedTaskNode` for the given `TaskNode`.
        If the task is a leaf, it calls `grade_leaf` to grade it.
        Otherwise, the task is graded by recursively grading its descendants bottom-up.
        """
        try:
            if current_depth >= self.max_depth and not task.is_leaf():
                logger.info(f"Max depth reached for task {task.id}. Approximating entire subtree.")
                return await self.grade_subtree(task)
            elif task.is_leaf():
                return await self._grade_leaf_with_retries(task, grade_leaf_fn)
        except openai.RateLimitError as e:
            logger.exception(f"Rate limit error while grading leaf {task.id}: {e}")
            raise
        except Exception as e:
            logger.exception(f"Grading leaf {task.id} failed!\n{e}")
            return GradedTaskNode.from_task(
                task,
                score=0.0,
                valid_score=False,
                explanation=str(e),
                judge_metadata=None,
            )

        graded_sub_tasks = await asyncio.gather(
            *(self.grade(t, grade_leaf_fn, current_depth + 1) for t in task.sub_tasks)
        )
        weighted_score = score_from_children(graded_sub_tasks)

        return GradedTaskNode(
            id=task.id,
            requirements=task.requirements,
            weight=task.weight,
            sub_tasks=graded_sub_tasks,
            score=weighted_score,
            valid_score=True,
            explanation="Aggregated score from sub-tasks.",
            judge_metadata=None,
        )

    async def _grade_leaf_with_retries(
        self,
        task: TaskNode,
        grade_leaf_fn: Callable[[TaskNode], Awaitable[GradedTaskNode]],
    ) -> GradedTaskNode:
        """Retry transient invalid leaf results without recomputing valid checkpoints."""
        for retry_number in range(self.max_invalid_leaf_retries + 1):
            try:
                graded = await grade_leaf_fn(task)
            except openai.RateLimitError:
                raise
            except openai.APITimeoutError as error:
                logger.warning(
                    "Leaf grading timed out; recording an invalid result without retrying",
                    task_id=task.id,
                    error=str(error),
                )
                return GradedTaskNode.from_task(
                    task,
                    score=0.0,
                    valid_score=False,
                    explanation=str(error),
                    judge_metadata=None,
                )
            except Exception as error:
                if retry_number >= self.max_invalid_leaf_retries:
                    raise
                logger.warning(
                    "Leaf grading failed; retrying",
                    task_id=task.id,
                    retry_number=retry_number + 1,
                    max_retries=self.max_invalid_leaf_retries,
                    error=str(error),
                )
                continue

            if graded.valid_score or retry_number >= self.max_invalid_leaf_retries:
                return graded
            logger.warning(
                "Leaf grading returned an invalid result; retrying",
                task_id=task.id,
                retry_number=retry_number + 1,
                max_retries=self.max_invalid_leaf_retries,
            )

        raise AssertionError("unreachable leaf retry state")

    @abstractmethod
    async def grade_leaf(self, task: TaskNode) -> GradedTaskNode:
        """Grades a leaf task (to be implemented by sub-classes)."""

        raise NotImplementedError()

    @abstractmethod
    async def grade_subtree(self, task: TaskNode) -> GradedTaskNode:
        """Approximates the grade for an entire subtree when the max depth is reached."""
        raise NotImplementedError()

    def get_logger(self, task: TaskNode) -> BoundLogger:
        """Creates a logger for a specific task."""

        if not self.log_path:
            _logger = logging.getLogger("null_logger")
            _logger.addHandler(logging.NullHandler())
            return wrap_logger(_logger)

        run_logger = logging.getLogger(task.id)
        run_logger.setLevel(logging.DEBUG)
        log_file_handler = logging.FileHandler(self.log_path / f"{task.id}.log")
        run_logger.addHandler(log_file_handler)
        run_logger.propagate = False

        return wrap_logger(run_logger)

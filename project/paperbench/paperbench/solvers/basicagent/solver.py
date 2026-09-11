import asyncio
import time

import blobfile as bf
import structlog
from openai import LengthFinishReasonError
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionUserMessageParam,
)
from openai.types.responses.web_search_tool_param import WebSearchToolParam
from typing_extensions import override

import chz
from nanoeval.solvers.computer_tasks.code_execution_interface import (
    ComputerInterface,
)
from paperbench.constants import AGENT_DIR_CONFIG
from paperbench.nano.structs import AgentOutput
from paperbench.nano.task import PBTask
from paperbench.solvers.base import BasePBSolver
from paperbench.solvers.basicagent.api import make_completer_request
from paperbench.solvers.basicagent.completer import (
    BasicAgentTurnCompleterConfig,
    OpenAIResponsesTurnCompleterConfig,
)
from paperbench.solvers.basicagent.logging import LoggableMessages
from paperbench.solvers.basicagent.prompts.templates import (
    DEFAULT_CONTINUE_MESSAGE,
    get_system_message,
)
from paperbench.solvers.basicagent.tools import (
    BashTool,
    PythonTool,
    ReadFileChunk,
    SearchFile,
    SubmitTool,
)
from paperbench.solvers.basicagent.tools.base import Tool
from paperbench.solvers.basicagent.utils import (
    cancel_async_task,
    format_progress_time,
    get_instructions,
    handle_sonnet_limits,
    handle_tool_call,
    optionally_upload_heavy_logs,
    prune_messages,
)
from paperbench.solvers.completion_review import (
    DEFAULT_MIN_REMAINING_SECONDS,
    MAX_CONSECUTIVE_QUICK_UNCHANGED,
    QUICK_UNCHANGED_MAX_SECONDS,
    CompletionReviewMode,
    build_completion_review_prompt,
    build_execution_feedback_command,
    build_execution_feedback_prompt,
    execution_feedback_paths,
    execution_feedback_timeout_seconds,
    get_submission_git_head,
    is_execution_feedback_infrastructure_error,
    is_quick_unchanged_review,
    remaining_budget_seconds,
    snapshot_initial_submission,
    write_completion_review_metadata,
)
from paperbench.solvers.upload import (
    start_periodic_light_log_upload,
    upload_heavy_logs,
)
from paperbench.solvers.utils import (
    check_for_existing_run,
    log_messages_to_file,
    sanity_check_docker,
)

logger = structlog.stdlib.get_logger(component=__name__)


@chz.chz
class BasicAgentSolver(BasePBSolver):
    completer_config: BasicAgentTurnCompleterConfig = chz.field(
        default_factory=lambda: OpenAIResponsesTurnCompleterConfig(
            model="gpt-4.1-mini",
            tools=[WebSearchToolParam(type="web_search_preview")],
        )
    )
    max_steps: int | None = chz.field(default=None)
    time_limit: int | None = chz.field(default=5 * 60, doc="Time limit in seconds (5min default)")
    use_submit_tool: bool = chz.field(default=True)
    use_real_time_limit: bool = chz.field(
        default=True,
        doc="If True, don't count API retrying time towards time limit",
    )
    iterative_agent: bool = chz.field(default=False)
    completion_review: bool = chz.field(
        default=False,
        doc="Continue after Submit calls while enough original budget remains",
    )
    completion_review_min_remaining_seconds: int = chz.field(
        default=DEFAULT_MIN_REMAINING_SECONDS,
        doc="Minimum original rollout budget required before starting the review",
    )
    completion_review_mode: CompletionReviewMode = chz.field(
        default="generic",
        doc="Use a neutral review or first expose a harness-run reproduce.sh log",
    )
    execution_feedback_timeout_seconds: int = chz.field(
        default=60 * 60,
        doc="Maximum reproduce.sh runtime before each execution-log review",
    )

    upload_interval_messages: int | None = chz.field(default=None)
    upload_interval_seconds: int | None = chz.field(default=1800)

    reminder_freq: int = chz.field(default=5, doc="Every how many steps to remind agent of state")

    @override
    def shortname(self) -> str:
        return "basicagent"

    def _get_tools(self) -> list[Tool]:
        if self.iterative_agent:
            tools = [BashTool(), ReadFileChunk()]
        else:
            tools = [BashTool(), PythonTool(), SearchFile(), ReadFileChunk()]
        if self.use_submit_tool:
            tools.append(SubmitTool())
        return tools

    async def _execute_agent(
        self,
        computer: ComputerInterface,
        task: PBTask,
        prompt: list[ChatCompletionMessageParam],
        start_time: int,
    ) -> int:
        """
        Executes the agent on the given prompt:
        - Stops when time limit is reached, messgae limit is reached, or submit tool is used
        - The message history is logged to task.run_dir / agent.log on every change
            to the message history
        - Periodically uploads the "heavy" logs
        - Prunes context window
        """
        ctx_logger = logger.bind(
            run_group_id=task.run_group_id, run_id=task.run_id, runs_dir=task.runs_dir
        )
        if self.completion_review_mode not in {"generic", "execution-log"}:
            raise ValueError(f"Unsupported completion_review_mode: {self.completion_review_mode}")
        if self.execution_feedback_timeout_seconds <= 0:
            raise ValueError("execution_feedback_timeout_seconds must be positive")

        tools = self._get_tools()
        self.completer_config.basicagent_tools = tools
        completer = self.completer_config.build()
        messages = LoggableMessages(
            init=prompt, log_path=bf.join(task.run_dir, "agent.log"), logger=log_messages_to_file
        )

        num_steps = 0
        total_retry_time = 0.0
        last_time_uploaded = time.time()
        review_count = 0
        review_iterations: list[dict[str, object]] = []
        initial_submission: str | None = None
        consecutive_quick_unchanged = 0

        upload_task = None
        try:
            while not self._has_agent_finished(
                num_steps=num_steps, start_time=start_time, total_retry_time=total_retry_time
            ):
                num_steps += 1
                optional_upload_outcome = await optionally_upload_heavy_logs(
                    computer=computer,
                    task=task,
                    run_dir=task.run_dir,
                    num_steps=num_steps,
                    start_time=start_time,
                    last_time_uploaded=last_time_uploaded,
                    upload_interval_messages=self.upload_interval_messages,
                    upload_interval_seconds=self.upload_interval_seconds,
                )
                last_time_uploaded, upload_task = (
                    optional_upload_outcome.last_time_uploaded,
                    optional_upload_outcome.upload_task,
                )

                # sonnet has a 1000 message limit we have to special case
                messages = handle_sonnet_limits(self.completer_config, messages)

                # Send reminder every self.reminder_freq steps
                if num_steps % self.reminder_freq == 0:
                    messages.append(self._construct_periodic_reminder(start_time, total_retry_time))

                # generate response
                try:
                    res = await make_completer_request(completer=completer, messages=messages.data)
                    (response_messages, tool_calls, response_retry_time) = (
                        res.response_messages,
                        res.tool_calls,
                        res.time_spent_retrying,
                    )
                    total_retry_time += response_retry_time
                except (LengthFinishReasonError, IndexError) as e:
                    ctx_logger.info(
                        "Context length exceeded. Pruning messages...", destinations=["run"]
                    )
                    prune_individual = "PRUNE_INDIVIDUAL_MESSAGES" in str(e)
                    messages = prune_messages(messages, prune_individual=prune_individual)
                    continue
                messages += response_messages

                if tool_calls:
                    for tool_call in tool_calls:
                        handled = await handle_tool_call(tool_call, tools, task, computer)
                        if handled is None:  # signal that submit was called; exit or review.
                            if self.completion_review:
                                if self.time_limit is None:
                                    raise ValueError(
                                        "completion_review requires a finite time_limit"
                                    )
                                decision_time = time.time()
                                if review_iterations:
                                    previous_review = review_iterations[-1]
                                    head_after = await get_submission_git_head(computer)
                                    quick_unchanged = is_quick_unchanged_review(
                                        head_before=previous_review.get("git_head_before")
                                        if isinstance(previous_review.get("git_head_before"), str)
                                        else None,
                                        head_after=head_after,
                                        started_at=float(previous_review["started_at"]),
                                        finished_at=decision_time,
                                    )
                                    consecutive_quick_unchanged = (
                                        consecutive_quick_unchanged + 1 if quick_unchanged else 0
                                    )
                                    previous_review.update(
                                        {
                                            "finished_at": decision_time,
                                            "completion_reason": "submit",
                                            "duration_seconds": decision_time
                                            - float(previous_review["started_at"]),
                                            "git_head_after": head_after,
                                            "quick_unchanged": quick_unchanged,
                                            "consecutive_quick_unchanged": (
                                                consecutive_quick_unchanged
                                            ),
                                        }
                                    )
                                    if (
                                        consecutive_quick_unchanged
                                        >= MAX_CONSECUTIVE_QUICK_UNCHANGED
                                    ):
                                        write_completion_review_metadata(
                                            task.run_dir,
                                            {
                                                "enabled": True,
                                                "performed": True,
                                                "mode": self.completion_review_mode,
                                                "budget_policy": "shared-agent-and-diagnostic-wall-clock",
                                                "minimum_remaining_seconds": self.completion_review_min_remaining_seconds,
                                                "quick_unchanged_max_seconds": QUICK_UNCHANGED_MAX_SECONDS,
                                                "max_consecutive_quick_unchanged": MAX_CONSECUTIVE_QUICK_UNCHANGED,
                                                "remaining_seconds_at_decision": remaining_budget_seconds(
                                                    time_limit_seconds=self.time_limit,
                                                    start_time=start_time,
                                                    now=decision_time,
                                                ),
                                                "initial_submission": initial_submission,
                                                "review_count": review_count,
                                                "iterations": review_iterations,
                                                "finished_at": decision_time,
                                                "completion_reason": "consecutive-quick-unchanged",
                                                "consecutive_quick_unchanged": consecutive_quick_unchanged,
                                            },
                                        )
                                        return num_steps
                                remaining_seconds = remaining_budget_seconds(
                                    time_limit_seconds=self.time_limit,
                                    start_time=start_time,
                                    now=decision_time,
                                )
                                if (
                                    remaining_seconds
                                    >= self.completion_review_min_remaining_seconds
                                ):
                                    await upload_heavy_logs(
                                        computer=computer,
                                        agent_start_time=int(start_time),
                                        agent_dir_config=AGENT_DIR_CONFIG,
                                        run_dir=task.run_dir,
                                        run_group_id=task.run_group_id,
                                        runs_dir=task.runs_dir,
                                        run_id=task.run_id,
                                        num_messages=num_steps,
                                        runtime=time.time() - start_time,
                                    )
                                    if initial_submission is None:
                                        initial_submission = snapshot_initial_submission(
                                            task.run_dir
                                        )
                                    execution_feedback: dict[str, object] | None = None
                                    if self.completion_review_mode == "execution-log":
                                        diagnostic_timeout = execution_feedback_timeout_seconds(
                                            remaining_seconds=remaining_seconds,
                                            minimum_review_seconds=(
                                                self.completion_review_min_remaining_seconds
                                            ),
                                            maximum_execution_seconds=(
                                                self.execution_feedback_timeout_seconds
                                            ),
                                        )
                                        if diagnostic_timeout <= 0:
                                            write_completion_review_metadata(
                                                task.run_dir,
                                                {
                                                    "enabled": True,
                                                    "performed": review_count > 0,
                                                    "mode": self.completion_review_mode,
                                                    "budget_policy": "shared-agent-and-diagnostic-wall-clock",
                                                    "minimum_remaining_seconds": self.completion_review_min_remaining_seconds,
                                                    "execution_feedback_timeout_seconds": self.execution_feedback_timeout_seconds,
                                                    "remaining_seconds_at_decision": remaining_seconds,
                                                    "initial_submission": initial_submission,
                                                    "review_count": review_count,
                                                    "iterations": review_iterations,
                                                    "completion_reason": "insufficient-budget-for-execution-feedback",
                                                },
                                            )
                                            return num_steps
                                        feedback_index = review_count + 1
                                        feedback_log_path, _ = execution_feedback_paths(
                                            feedback_index
                                        )
                                        feedback_started_at = time.time()
                                        feedback_result = await computer.send_shell_command(
                                            build_execution_feedback_command(
                                                iteration=feedback_index,
                                                timeout_seconds=diagnostic_timeout,
                                            )
                                        )
                                        feedback_finished_at = time.time()
                                        execution_feedback = {
                                            "timeout_seconds": diagnostic_timeout,
                                            "log_path": feedback_log_path,
                                            "started_at": feedback_started_at,
                                            "finished_at": feedback_finished_at,
                                            "duration_seconds": (
                                                feedback_finished_at - feedback_started_at
                                            ),
                                            "exit_code": feedback_result.exit_code,
                                            "timed_out": feedback_result.exit_code == 124,
                                            "error": None,
                                        }
                                        if is_execution_feedback_infrastructure_error(
                                            feedback_result.exit_code
                                        ):
                                            write_completion_review_metadata(
                                                task.run_dir,
                                                {
                                                    "enabled": True,
                                                    "performed": review_count > 0,
                                                    "mode": self.completion_review_mode,
                                                    "budget_policy": "shared-agent-and-diagnostic-wall-clock",
                                                    "minimum_remaining_seconds": self.completion_review_min_remaining_seconds,
                                                    "execution_feedback_timeout_seconds": self.execution_feedback_timeout_seconds,
                                                    "remaining_seconds_at_decision": remaining_seconds,
                                                    "initial_submission": initial_submission,
                                                    "review_count": review_count,
                                                    "iterations": review_iterations,
                                                    "completion_reason": "execution-feedback-infrastructure-error",
                                                },
                                            )
                                            return num_steps
                                        remaining_seconds = remaining_budget_seconds(
                                            time_limit_seconds=self.time_limit,
                                            start_time=start_time,
                                            now=time.time(),
                                        )
                                        if (
                                            remaining_seconds
                                            < self.completion_review_min_remaining_seconds
                                        ):
                                            write_completion_review_metadata(
                                                task.run_dir,
                                                {
                                                    "enabled": True,
                                                    "performed": review_count > 0,
                                                    "mode": self.completion_review_mode,
                                                    "budget_policy": "shared-agent-and-diagnostic-wall-clock",
                                                    "minimum_remaining_seconds": self.completion_review_min_remaining_seconds,
                                                    "execution_feedback_timeout_seconds": self.execution_feedback_timeout_seconds,
                                                    "remaining_seconds_at_decision": remaining_seconds,
                                                    "initial_submission": initial_submission,
                                                    "review_count": review_count,
                                                    "iterations": review_iterations,
                                                    "completion_reason": "insufficient-original-budget",
                                                },
                                            )
                                            return num_steps
                                        review_prompt = build_execution_feedback_prompt(
                                            remaining_seconds=remaining_seconds,
                                            log_path=feedback_log_path,
                                            reproduction_timeout_seconds=(diagnostic_timeout),
                                        )
                                    else:
                                        review_prompt = build_completion_review_prompt(
                                            remaining_seconds
                                        )
                                    head_before = await get_submission_git_head(computer)
                                    messages.append(
                                        {
                                            "role": "tool",
                                            "content": "Completion recorded; continue with the review.",
                                            "tool_call_id": tool_call.call_id,
                                        }
                                    )
                                    messages.append({"role": "user", "content": review_prompt})
                                    review_count += 1
                                    review_iterations.append(
                                        {
                                            "index": review_count,
                                            "prompt": review_prompt,
                                            "remaining_seconds_at_start": remaining_seconds,
                                            "started_at": time.time(),
                                            "git_head_before": head_before,
                                            **(
                                                {"execution_feedback": (execution_feedback)}
                                                if execution_feedback is not None
                                                else {}
                                            ),
                                        }
                                    )
                                    write_completion_review_metadata(
                                        task.run_dir,
                                        {
                                            "enabled": True,
                                            "performed": True,
                                            "mode": self.completion_review_mode,
                                            "budget_policy": "shared-agent-and-diagnostic-wall-clock",
                                            "minimum_remaining_seconds": self.completion_review_min_remaining_seconds,
                                            "remaining_seconds_at_start": review_iterations[0][
                                                "remaining_seconds_at_start"
                                            ],
                                            "started_at": review_iterations[0]["started_at"],
                                            "initial_submission": initial_submission,
                                            "prompt": review_prompt,
                                            "review_count": review_count,
                                            "iterations": review_iterations,
                                            "quick_unchanged_max_seconds": QUICK_UNCHANGED_MAX_SECONDS,
                                            "max_consecutive_quick_unchanged": MAX_CONSECUTIVE_QUICK_UNCHANGED,
                                            "consecutive_quick_unchanged": consecutive_quick_unchanged,
                                        },
                                    )
                                    break
                                completion_metadata: dict[str, object] = {
                                    "enabled": True,
                                    "performed": review_count > 0,
                                    "mode": self.completion_review_mode,
                                    "budget_policy": "shared-agent-and-diagnostic-wall-clock",
                                    "minimum_remaining_seconds": self.completion_review_min_remaining_seconds,
                                    "remaining_seconds_at_decision": remaining_seconds,
                                    "initial_submission": initial_submission,
                                    "review_count": review_count,
                                    "iterations": review_iterations,
                                }
                                if review_count:
                                    completion_metadata.update(
                                        {
                                            "finished_at": decision_time,
                                            "completion_reason": ("insufficient-original-budget"),
                                        }
                                    )
                                else:
                                    completion_metadata["skip_reason"] = (
                                        "insufficient-original-budget"
                                    )
                                write_completion_review_metadata(
                                    task.run_dir,
                                    completion_metadata,
                                )
                            return num_steps
                        messages.append(handled)
                else:
                    messages.append({"role": "user", "content": DEFAULT_CONTINUE_MESSAGE})
        except asyncio.TimeoutError as e:
            # gracefully handle upload_task, then re-raise
            if upload_task:
                await asyncio.shield(upload_task)
            raise e

        return num_steps

    def _has_agent_finished(
        self, num_steps: int, start_time: float, total_retry_time: float
    ) -> bool:
        """
        Returns True if the agent has reached the step limit or the time limit.
        """
        over_step_limit = self.max_steps is not None and num_steps > self.max_steps

        if self.use_real_time_limit and self.time_limit:
            over_time_limit = time.time() - start_time - total_retry_time > self.time_limit
        elif self.time_limit:
            over_time_limit = time.time() - start_time > self.time_limit
        else:
            over_time_limit = False

        return over_step_limit or over_time_limit

    def _construct_periodic_reminder(
        self, start_time: float, total_retry_time: float
    ) -> ChatCompletionUserMessageParam:
        """Constructs a message periodically reminding the agent of various tips."""
        if self.use_real_time_limit and self.time_limit:
            elapsed_time = time.time() - start_time - total_retry_time
            periodic_msg = f"Info: {format_progress_time(elapsed_time)} time elapsed out of {format_progress_time(self.time_limit)}. Remember, you only have to stop working when the time limit has been reached."
        if self.time_limit:
            elapsed_time = time.time() - start_time
            periodic_msg = f"Info: {format_progress_time(elapsed_time)} time elapsed out of {format_progress_time(self.time_limit)}. Remember, you only have to stop working when the time limit has been reached."
        else:
            elapsed_time = time.time() - start_time
            periodic_msg = f"Info: {format_progress_time(elapsed_time)} time elapsed"
        periodic_msg += "\n\nNote: Don't forget to git commit regularly!"
        message: ChatCompletionUserMessageParam = {
            "role": "user",
            "content": periodic_msg,
        }
        return message

    async def _execute_agent_and_periodically_upload_logs(
        self,
        computer: ComputerInterface,
        task: PBTask,
        prompt: list[ChatCompletionMessageParam],
    ) -> None:
        """
        Starts agent execution and periodically uploads the "light" logs. When agent finishes, it
        cancels periodic uploading, then performs one final upload of the "heavy" logs.
        """
        ctx_logger = logger.bind(
            run_group_id=task.run_group_id,
            run_id=task.run_id,
            runs_dir=task.runs_dir,
            destinations=["run"],
        )

        num_steps = -1
        start_time = int(time.time())
        agent_task: asyncio.Task[int] | None = None
        light_periodic_upload_task: asyncio.Task[None] | None = None

        try:
            async with asyncio.timeout(self.time_limit):
                light_periodic_upload_task = await start_periodic_light_log_upload(
                    agent_start_time=start_time,
                    run_dir=task.run_dir,
                    run_group_id=task.run_group_id,
                    runs_dir=task.runs_dir,
                    run_id=task.run_id,
                )

                agent_task = asyncio.create_task(
                    self._execute_agent(
                        computer=computer,
                        task=task,
                        prompt=prompt,
                        start_time=start_time,
                    )
                )

                while not agent_task.done():
                    ctx_logger.info("Waiting for agent to finish...")
                    if light_periodic_upload_task.done():
                        exc = light_periodic_upload_task.exception()
                        if exc:
                            raise exc
                    await asyncio.sleep(60)
                num_steps = await agent_task
        except asyncio.TimeoutError as e:
            ctx_logger.info(
                f"Agent run timed out after {time.time() - start_time} second (timeout: {self.time_limit}): {e}",
            )
        except asyncio.CancelledError as e:
            ctx_logger.info(
                f"Agent run cancelled after {time.time() - start_time} seconds: {e}",
            )
        finally:
            await cancel_async_task(async_task=agent_task, pb_task=task)
            await cancel_async_task(async_task=light_periodic_upload_task, pb_task=task)

        # one last upload
        await upload_heavy_logs(
            computer=computer,
            agent_start_time=int(start_time),
            agent_dir_config=AGENT_DIR_CONFIG,
            run_dir=task.run_dir,
            run_group_id=task.run_group_id,
            runs_dir=task.runs_dir,
            run_id=task.run_id,
            num_messages=num_steps,
        )

    async def _run_agent(self, computer: ComputerInterface, task: PBTask) -> AgentOutput:
        """
        Prepares agent, runs agent, then returns agent output. Skips agent run if we already have agent logs
        for this task.
        """
        ctx_logger = logger.bind(
            run_group_id=task.run_group_id, run_id=task.run_id, runs_dir=task.runs_dir
        )

        # Don't run the agent if we already have agent logs
        agent_output = await check_for_existing_run(task)
        if agent_output:
            return agent_output

        await sanity_check_docker(computer)

        start_time = time.time()
        instructions = await get_instructions(computer, task, self.time_limit)
        system_message = get_system_message(self.iterative_agent, task.judge.code_only)
        await self._execute_agent_and_periodically_upload_logs(
            computer=computer,
            task=task,
            prompt=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": instructions},
            ],
        )

        ctx_logger.info(
            f"Agent `{self.shortname()}` finished running for `{task.question_id}.{task.attempt_id}`!",
            destinations=["group", "run"],
            _print=True,
        )

        return AgentOutput(
            run_id=task.run_id,
            time_start=start_time,
            time_end=time.time(),
            error_msg=None,
            runtime_in_seconds=time.time() - start_time,
            status_exists=bf.exists(bf.join(task.run_dir, "status.json")),
        )

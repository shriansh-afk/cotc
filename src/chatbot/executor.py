"""DAG execution engine with parallel execution and fault tolerance."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from .checkpoint import CheckpointManager
from .dead_letter import DeadLetterQueue
from .decomposition import DecompositionEngine
from .llm import LLMProvider
from .models import TaskDAG, TaskNode, TaskStatus, ToolCallRecord
from .tools import ToolRegistry, create_default_registry

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of executing a single task."""

    success: bool
    value: str | None = None
    error: str | None = None


@dataclass
class ExecutionConfig:
    """Configuration for DAG execution."""

    max_retries: int = 3
    base_retry_delay: float = 1.0
    task_timeout: float = 300.0
    max_parallel_tasks: int = 10
    enable_tools: bool = True
    
    # Tool call limits
    max_tool_calls_per_task: int | None = None  # None = unlimited
    soft_warning_threshold: int = 50  # Warn at 50, 100, 150, etc.
    
    # Loop detection
    enable_loop_detection: bool = True
    loop_detection_window: int = 3  # Check last 3 calls for identical patterns
    
    # Progress tracking
    enable_progress_tracking: bool = True
    progress_window: int = 5  # Check last 5 outputs for changes
    
    allowed_tools: list[str] | None = None  # None = all tools
    skip_simplicity_check: bool = False  # Allow bypassing simplicity check for testing
    ask_for_help_on_failure: bool = True  # Ask user for help when critical tasks fail



class DAGExecutor:
    """Executes task DAGs with parallel execution and fault tolerance."""

    def __init__(
        self,
        llm: LLMProvider,
        decomposition_engine: DecompositionEngine,
        checkpoint_manager: CheckpointManager,
        config: ExecutionConfig | None = None,
        tool_registry: ToolRegistry | None = None,
    ):
        self.llm = llm
        self.decomposition_engine = decomposition_engine
        self.checkpoint_manager = checkpoint_manager
        self.config = config or ExecutionConfig()
        self.dead_letter_queue = DeadLetterQueue()
        self.tool_registry = tool_registry or create_default_registry()

    async def execute_dag(self, dag: TaskDAG) -> dict[str, Any]:
        """
        Execute all tasks in the DAG respecting dependencies.

        Args:
            dag: The TaskDAG to execute

        Returns:
            Dictionary with execution results and statistics
        """
        await self.checkpoint_manager.save_checkpoint(dag)

        completed: set[str] = set()
        failed: set[str] = set()
        skipped: set[str] = set()

        while not dag.all_tasks_done():
            ready_tasks = dag.get_ready_tasks()

            # Handle tasks blocked by upstream critical failures
            blocked_tasks = self._get_tasks_blocked_by_failures(dag)
            for task in blocked_tasks:
                task.status = TaskStatus.SKIPPED
                task.error = "Blocked: upstream task had critical tool failures"
                skipped.add(task.id)
                logger.warning(f"Task {task.id} skipped due to upstream critical failures")
                self._mark_dependents_skipped(dag, task.id, skipped)

            if not ready_tasks and not dag.all_tasks_done():
                logger.warning("Deadlock detected: no tasks ready but DAG not complete")
                break

            # Check for complex tasks that need decomposition
            tasks_to_execute = []
            for task in ready_tasks:
                if task.is_complex:
                    try:
                        was_decomposed = await self.decomposition_engine.maybe_decompose_task(
                            dag, task
                        )
                        if was_decomposed:
                            await self.checkpoint_manager.save_checkpoint(dag)
                            continue
                    except ValueError as e:
                        logger.error(f"Decomposition error for task {task.id}: {e}")
                        task.is_complex = False

                tasks_to_execute.append(task)

            if not tasks_to_execute:
                continue

            # Execute ready tasks in parallel
            logger.warning(f"[EXEC] Executing {len(tasks_to_execute)} task(s): {', '.join(t.name or t.id for t in tasks_to_execute)}")
            results = await self._parallel_execute(tasks_to_execute, dag)

            for task, result in results:
                if result.success:
                    task.status = TaskStatus.COMPLETED
                    task.result = result.value
                    completed.add(task.id)
                    logger.warning(f"[OK] Completed: {task.name or task.id}")
                else:
                    logger.warning(f"[FAIL] Failed: {task.name or task.id} - {result.error or 'Unknown error'}")
                    await self._handle_failure(dag, task, result.error or "Unknown error", failed, skipped)

            await self.checkpoint_manager.save_checkpoint(dag)

        return {
            "completed": completed,
            "failed": failed,
            "skipped": skipped,
            "stats": dag.get_completion_stats(),
        }

    async def _parallel_execute(
        self, tasks: list[TaskNode], dag: TaskDAG
    ) -> list[tuple[TaskNode, ExecutionResult]]:
        """Execute multiple tasks in parallel."""
        semaphore = asyncio.Semaphore(self.config.max_parallel_tasks)

        async def execute_with_semaphore(task: TaskNode) -> tuple[TaskNode, ExecutionResult]:
            async with semaphore:
                task.status = TaskStatus.RUNNING
                result = await self._execute_single_task(task, dag)
                return task, result

        coroutines = [execute_with_semaphore(task) for task in tasks]
        results = await asyncio.gather(*coroutines, return_exceptions=True)

        processed_results = []
        for i, result in enumerate(results):
            task = tasks[i]
            if isinstance(result, Exception):
                processed_results.append(
                    (task, ExecutionResult(success=False, error=str(result)))
                )
            else:
                processed_results.append(result)

        return processed_results

    async def _execute_single_task(self, task: TaskNode, dag: TaskDAG) -> ExecutionResult:
        """Execute a single task with optional tool support."""
        try:
            # Build prompt with dependency results
            prompt = self._build_task_prompt(task, dag)
            system_prompt = (
                "You are an autonomous AI agent completing a research sub-task. "
                "Your goal is to complete tasks end-to-end without asking the user for clarification. "
                "Make reasonable assumptions when details are ambiguous and proceed with the most logical approach. "
                "Focus on delivering complete, actionable results."
            )

            # If tools are disabled, use simple call
            if not self.config.enable_tools:
                response = await asyncio.wait_for(
                    self.llm.call(prompt=prompt, system_prompt=system_prompt),
                    timeout=self.config.task_timeout,
                )
                return ExecutionResult(success=True, value=response)

            # Tool-enabled execution with message loop
            messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
            tools = self.tool_registry.get_openai_schema(self.config.allowed_tools)

            if self.config.enable_tools:
                system_prompt += (
                    "\n\nYou have access to tools that can help you complete tasks. "
                    "Use them when appropriate to gather information, perform calculations, "
                    "or interact with files."
                    "\n\n**Important Guidelines:**"
                    "\n- NEVER ask the user for clarification or preferences. Make reasonable assumptions."
                    "\n- When faced with multiple options, choose the most common or widely-used approach."
                    "\n- Complete the entire task from start to finish using available tools."
                    "\n- Provide concrete results, not just plans or suggestions."
                    "\n\n**Handling Missing Packages:**"
                    "\n- If you encounter ImportError or ModuleNotFoundError when using python_execute, "
                    "use the pip_install tool to install the missing package(s)."
                    "\n- After successful installation, retry the original operation."
                    "\n- Example: If you get 'ModuleNotFoundError: No module named requests', "
                    "call pip_install with packages=['requests'], then retry your code."
                    "\n- For packages with specific versions, include them: packages=['requests>=2.28.0']"
                    "\n- Only report errors if pip_install fails after attempting installation."
                )

            tool_call_count = 0

            while True:
                # Soft warning at thresholds (50, 100, 150, etc.)
                if (self.config.soft_warning_threshold and 
                    tool_call_count > 0 and 
                    tool_call_count % self.config.soft_warning_threshold == 0):
                    logger.warning(
                        f"Task {task.id} has made {tool_call_count} tool calls "
                        f"(soft warning threshold: {self.config.soft_warning_threshold})"
                    )
                
                # Hard limit check (if configured)
                if self.config.max_tool_calls_per_task is not None:
                    if tool_call_count >= self.config.max_tool_calls_per_task:
                        logger.warning(
                            f"Task {task.id} reached max tool calls ({self.config.max_tool_calls_per_task})"
                        )
                        break
                
                # Intelligent loop detection
                if self._detect_repeated_calls(task):
                    return ExecutionResult(
                        success=False,
                        error=f"Infinite loop detected: same tool call repeated {self.config.loop_detection_window} times"
                    )
                
                # Progress tracking
                if not self._is_making_progress(task):
                    return ExecutionResult(
                        success=False,
                        error=f"Task stuck: no progress in last {self.config.progress_window} tool calls"
                    )

                # Call LLM with tools
                response = await asyncio.wait_for(
                    self.llm.call_with_tools(
                        messages=messages,
                        tools=tools,
                        system_prompt=system_prompt,
                    ),
                    timeout=self.config.task_timeout,
                )

                # If no tool calls, we're done
                if not response.tool_calls:
                    result_value = response.content or ""

                    # Check if task has critical tool failures and set the flag
                    if task.has_critical_tool_failures():
                        task.has_tool_failures = True
                        failed_tools = [
                            tc.name for tc in task.tool_calls
                            if tc.result and not tc.result.get("success", False)
                        ]
                        failure_note = f"\n\n[WARNING: Critical tool failures occurred. Failed tools: {', '.join(failed_tools)}]"
                        result_value += failure_note

                    return ExecutionResult(success=True, value=result_value)

                # Add assistant message with tool calls
                assistant_msg: dict[str, Any] = {"role": "assistant", "content": response.content}
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": str(tc.arguments)},
                    }
                    for tc in response.tool_calls
                ]
                messages.append(assistant_msg)

                # Execute each tool call
                for tool_call in response.tool_calls:
                    tool_call_count += 1
                    
                    # Log important tool calls to terminal
                    if tool_call.name == "pip_install":
                        logger.warning(f"  [PKG] Installing packages: {tool_call.arguments.get('packages', [])}")
                    elif tool_call.name in ["python_execute", "web_download", "web_search"]:
                        logger.info(f"  [TOOL] Using tool: {tool_call.name}")

                    result = await self.tool_registry.execute(
                        tool_call.name, tool_call.arguments
                    )

                    # Record the tool call in task history
                    task.tool_calls.append(
                        ToolCallRecord(
                            id=tool_call.id,
                            name=tool_call.name,
                            arguments=tool_call.arguments,
                            result={
                                "success": result.success,
                                "output": result.output,
                                "error": result.error,
                            },
                        )
                    )

                    # Add tool result message
                    result_content = (
                        str(result.output) if result.success else f"Error: {result.error}"
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result_content,
                        }
                    )

                # Continue loop - LLM will process tool results

            # If we exit the loop due to max tool calls, return what we have
            result_value = response.content or "Task completed (max tool calls reached)"

            # Check if task has critical tool failures and set the flag
            if task.has_critical_tool_failures():
                task.has_tool_failures = True
                failed_tools = [
                    tc.name for tc in task.tool_calls
                    if tc.result and not tc.result.get("success", False)
                ]
                failure_note = f"\n\n[WARNING: Critical tool failures occurred. Failed tools: {', '.join(failed_tools)}]"
                result_value += failure_note

            return ExecutionResult(success=True, value=result_value)

        except asyncio.TimeoutError:
            return ExecutionResult(success=False, error="Task execution timed out")
        except Exception as e:
            return ExecutionResult(success=False, error=str(e))

    def _build_tool_summary(self, task: TaskNode) -> str:
        """Build a summary of tool call results for context."""
        if not task.tool_calls:
            return ""

        failed = [tc for tc in task.tool_calls if tc.result and not tc.result.get("success")]
        if not failed:
            return ""

        lines = ["[Tool Call Summary]"]
        lines.append(f"- FAILED: {', '.join(tc.name for tc in failed)}")
        for tc in failed:
            error = tc.result.get("error", "Unknown") if tc.result else "Unknown"
            lines.append(f"  - {tc.name}: {error}")
        return "\n".join(lines)

    def _detect_repeated_calls(self, task: TaskNode) -> bool:
        """Detect if the same tool call is being repeated with identical arguments.
        
        Returns True if the last N tool calls are identical (same name and args),
        indicating a potential infinite loop.
        """
        if not self.config.enable_loop_detection:
            return False
            
        if len(task.tool_calls) < self.config.loop_detection_window:
            return False
        
        recent = task.tool_calls[-self.config.loop_detection_window:]
        first = recent[0]
        
        # Check if all recent calls have same name and arguments
        for call in recent[1:]:
            if call.name != first.name or call.arguments != first.arguments:
                return False
        
        # All calls in window are identical
        logger.warning(
            f"Loop detected in task {task.id}: "
            f"'{first.name}' called {self.config.loop_detection_window} times with same arguments"
        )
        return True

    def _is_making_progress(self, task: TaskNode) -> bool:
        """Check if recent tool calls are producing different outputs.
        
        Returns False if the last N tool outputs are all identical,
        indicating the task is stuck and not making progress.
        """
        if not self.config.enable_progress_tracking:
            return True  # Progress tracking disabled
            
        if len(task.tool_calls) < self.config.progress_window:
            return True  # Not enough data yet
        
        recent = task.tool_calls[-self.config.progress_window:]
        
        # Extract outputs from successful tool calls
        outputs = []
        for tc in recent:
            if tc.result:
                output = str(tc.result.get("output", ""))
                # Normalize whitespace for comparison
                output = " ".join(output.split())
                outputs.append(output)
        
        if not outputs:
            return True  # No outputs to compare
        
        # If all outputs are identical, no progress is being made
        unique_outputs = set(outputs)
        if len(unique_outputs) == 1 and len(outputs) >= self.config.progress_window:
            logger.warning(
                f"No progress detected in task {task.id}: "
                f"last {self.config.progress_window} tool calls produced identical output"
            )
            return False
        
        return True


    def _build_retry_context(self, task: TaskNode) -> str:
        """Build retry context with error information from previous attempts."""
        if task.retry_count == 0:
            return ""

        lines = []
        lines.append(f"⚠️ RETRY ATTEMPT #{task.retry_count}")
        lines.append("")
        lines.append("Your previous attempt failed with the following error:")
        lines.append(f"{task.error}")
        lines.append("")

        # List failed tool calls
        failed_tools = [
            tc for tc in task.tool_calls
            if tc.result and not tc.result.get("success", False)
        ]

        if failed_tools:
            lines.append("Failed tool calls:")
            for tc in failed_tools:
                error_msg = tc.result.get("error", "Unknown error") if tc.result else "Unknown error"
                lines.append(f"  - {tc.name}: {error_msg}")
            lines.append("")

        lines.append("Please analyze what went wrong and correct your approach. Pay special attention to:")
        lines.append("  - Syntax errors in generated code (check for missing commas, quotes, brackets)")
        lines.append("  - Missing imports or dependencies (use pip_install if needed)")
        lines.append("  - Incorrect function arguments or data types")
        lines.append("  - JSON formatting issues")
        lines.append("")
        lines.append("---")
        lines.append("")

        return "\n".join(lines)

    def _build_task_prompt(self, task: TaskNode, dag: TaskDAG) -> str:
        """Build the prompt for a task, including dependency results."""
        prompt_parts = []

        # Add retry context if this is a retry attempt
        retry_context = self._build_retry_context(task)
        if retry_context:
            prompt_parts.append(retry_context)

        prompt_parts.append(task.prompt)

        # Add context from dependencies
        dep_context = []
        for dep_id in task.depends_on:
            dep_task = dag.get_task(dep_id)
            if dep_task and dep_task.result:
                # Include tool failure summary if the dependency had failures
                tool_summary = self._build_tool_summary(dep_task) if dep_task.has_tool_failures else ""
                context_text = f"## Result from '{dep_task.name or dep_id}':\n"
                if tool_summary:
                    context_text += f"{tool_summary}\n\n"
                context_text += dep_task.result
                dep_context.append(context_text)
            elif dep_task and dep_task.status == TaskStatus.SKIPPED:
                dep_context.append(f"## Result from '{dep_task.name or dep_id}':\n[UNAVAILABLE - task was skipped]")

        if dep_context:
            prompt_parts.insert(0, "Context from previous tasks:\n" + "\n\n".join(dep_context) + "\n\n---\n")

        return "\n".join(prompt_parts)

    async def _handle_failure(
        self,
        dag: TaskDAG,
        task: TaskNode,
        error: str,
        failed: set[str],
        skipped: set[str],
    ) -> None:
        """Handle a task failure with retry logic."""
        task.retry_count += 1
        task.error = error

        if task.retry_count <= self.config.max_retries:
            # Exponential backoff
            wait_time = self.config.base_retry_delay * (2 ** (task.retry_count - 1))
            logger.info(f"Task {task.id} failed, retrying in {wait_time}s (attempt {task.retry_count})")
            await asyncio.sleep(wait_time)
            task.status = TaskStatus.PENDING
        else:
            # Max retries exceeded
            logger.error(f"Task {task.id} failed after {self.config.max_retries} retries: {error}")
            task.status = TaskStatus.FAILED
            failed.add(task.id)

            # Add to dead letter queue
            self.dead_letter_queue.add(task, dag.id, error)

            # Mark dependents as skipped
            self._mark_dependents_skipped(dag, task.id, skipped)

    def _mark_dependents_skipped(
        self, dag: TaskDAG, failed_task_id: str, skipped: set[str]
    ) -> None:
        """Recursively mark tasks that depend on a failed task as skipped."""
        for task in dag.tasks.values():
            if failed_task_id in task.depends_on:
                if task.status == TaskStatus.PENDING:
                    task.status = TaskStatus.SKIPPED
                    skipped.add(task.id)
                    logger.info(f"Task {task.id} skipped due to failed dependency {failed_task_id}")
                    # Recursively skip dependents
                    self._mark_dependents_skipped(dag, task.id, skipped)

    def _get_tasks_blocked_by_failures(self, dag: TaskDAG) -> list[TaskNode]:
        """Get pending tasks blocked by dependencies with critical tool failures."""
        failed_completion_ids = {
            tid for tid, t in dag.tasks.items()
            if t.status == TaskStatus.COMPLETED and t.has_tool_failures
        }

        return [
            task for task in dag.tasks.values()
            if task.status == TaskStatus.PENDING
            and any(dep_id in failed_completion_ids for dep_id in task.depends_on)
        ]

    async def resume_from_checkpoint(self, dag_id: str) -> dict[str, Any] | None:
        """
        Resume execution from a checkpoint.

        Args:
            dag_id: The ID of the DAG to resume

        Returns:
            Execution results, or None if checkpoint not found
        """
        dag = await self.checkpoint_manager.load_checkpoint(dag_id)
        if dag is None:
            return None

        # Reset any running tasks to pending
        await self.checkpoint_manager.prepare_for_resume(dag)

        return await self.execute_dag(dag)

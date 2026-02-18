# Comparison: `cotc` vs `totc`

This document outlines the differences between the two AI agent implementations found in your workspace: `cotc` and `totc`.

## Executive Summary

- **`cotc` (Chain of Thought / Decomposed):** A **Fault-Tolerant Task Orchestrator**. It focuses on **upfront planning**, decomposing complex problems into a Directed Acyclic Graph (DAG) of dependencies, and executing them with reliability features like retries and parallel processing. It is engineered for stability and efficiency in completing defined tasks.
- **`totc` (Tree of Thought Chain):** A **Recursive Exploration Agent**. It focuses on **iterative reasoning**, dynamically spawning "sub-projects" (recursive calls) to solve problems as they arise. It is engineered for depth, exploration, and handling open-ended complex reasoning paths.

## 1. Architecture & Data Structure

| Feature | `cotc` | `totc` |
| :--- | :--- | :--- |
| **Core Structure** | **DAG (Directed Acyclic Graph)** | **Tree (Recursive Call Stack)** |
| **Planning Approach** | **Upfront Decomposition**: Breaks the entire problem into a list of tasks and dependencies *before* starting execution. | **Iterative / On-the-fly**: Decides the next step (Code, Sub-project, or Response) one at a time. |
| **State Management** | Centralized `TaskDAG` with explicit `TaskNode` states (PENDING, COMPLETED, FAILED). | Implicit call stack state; `CodeExecutor` state persists within a customized REPL loop. |
| **Decomposition** | Explicit. Uses a `DecompositionEngine` to split prompts into `min` number of sub-tasks. | Implicit. Uses `sub_project` tool to create new agent instances for sub-problems. |

## 2. Execution Model

### `cotc` (Parallel & Robust)
- **Parallelism:** Can execute multiple independent tasks simultaneously (`asyncio.gather`, `max_parallel_tasks`).
- **Dependency Management:** Tasks wait for their `depends_on` predecessors to complete.
- **Fault Tolerance:**
    - **Retries:** Configurable `max_retries` with exponential backoff.
    - **Dead Letter Queue:** Captures failed tasks for manual inspection or later reprocessing.
    - **Checkpoints:** Saves DAG state to disk to allow resuming interrupted runs.

### `totc` (Sequential & Deep)
- **Sequential/Recursive:** Executes steps one by one. If a `sub_project` is called, the parent waits for the child to finish (Depth-First Search approach).
- **Interactive:** Designed effectively as a REPL (Read-Eval-Print Loop) where the user can intervene, approve code, or modify the state.
- **Tree Logging:** Integrated `TreeLogger` to visualize the reasoning trace in real-time.

## 3. Code Implementation Highlights

### `cotc`
*Located in `src/chatbot/`*
- **`decomposition.py`**: Uses specific prompts to break user queries into JSON task lists. Includes "Complexity Checks" to decide if decomposition is even necessary.
- **`executor.py`**: Manages the `run` loop, checking for ready tasks, handling tool calls, and managing the `DeadLetterQueue`.
- **`models.py`**: Defines structured data models like `TaskDAG`, `TaskNode`, and `TaskStatus`.

### `totc`
*Located in `totc/`*
- **`chain_of_thought.py`**: Single-file core. Implements the `ChainOfThoughtAgent` class.
- **`run()` method**: The main recursive loop.
    ```python
    # Logic in totc
    while iteration < max_iterations:
        decision = llm.decide()
        if decision.tool == "sub_project":
            self.run(decision.content, depth + 1) # Recursive call
        elif decision.tool == "code":
            self.execute_code(decision.content)
    ```

## 4. When to Use Which?

| Use `cotc` when... | Use `totc` when... |
| :--- | :--- |
| You have a defined multifaceted task (e.g., "Research X, Y, and Z and write a report"). | You have an ambiguous problem requiring exploration (e.g., "Solve this math proof" or "Debug this error"). |
| You care about speed (parallelism). | You care about reasoning depth and step-by-step verification. |
| Reliability and recovering from crashes is critical. | You want to interactively guide the agent or visualize the thought process. |
| The tasks are relatively independent. | The solution path is unknown and requires branching logic. |

# Fault-Tolerant Chatbot

An autonomous AI agent designed to decompose complex questions into manageable sub-tasks, execute them in parallel, and recover from failures automatically.

## Description

This project implements a sophisticated chatbot system that goes beyond simple question-answering. It uses a Directed Acyclic Graph (DAG) approach to break down complex queries into dependent and independent tasks. The system features a robust execution engine that handles parallel processing, automatically manages dependencies, and includes self-healing capabilities such as automatic Python package installation when code execution fails.

## Key Features

- **Task Decomposition**: Automatically breaks down complex user queries into optimal sub-tasks.
- **Parallel Execution**: executes independent tasks concurrently to reduce total processing time.
- **Fault Tolerance**: Includes retry logic, dead letter queues for failed tasks, and checkpointing to resume interrupted sessions.
- **Self-Healing**: Automatically detects missing Python packages during code execution and installs them on-the-fly.
- **Tool Integration**: Built-in tools for Python code execution, web search, file operations, and web scraping.
- **State Management**: Persists execution state to allow resuming work after interruptions.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd cotc
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   
   If you do not have a requirements file, the core dependencies are:
   - openai
   - pydantic
   - redis (optional, for Redis storage)
   - duckduckgo-search (for web search)
   - beautifulsoup4 (for web scraping)

## Configuration

The system requires an OpenAI API key to function.

1. Create a `.env` file in the project root.
2. Add your API key:
   ```
   OPENAI_API_KEY=your-api-key-here
   ```

## Usage

### Command Line Interface

You can interact with the chatbot directly from the terminal.

**Single Question:**
```bash
python ask.py "Calculate the Fibonacci sequence up to 100 and plot the values."
```

**Interactive Mode:**
```bash
python ask.py
```
This will start a REPL session where you can ask multiple questions. Type `quit`, `exit`, or `q` to leave.

### Jupyter Notebook

Launch the included notebook for an interactive environment that is compatible with Google Colab:

```bash
jupyter notebook ask_colab.ipynb
```

This notebook provides a complete setup cell to install dependencies and configure the environment, making it easy to test the agent's capabilities.

## Architecture

The system is organized into several key modules within `src/chatbot`:

- **api.py**: High-level interface for the chatbot.
- **decomposition.py**: Components for breaking down questions into task DAGs.
- **executor.py**: The engine responsible for executing the DAG, handling parallelism and retries.
- **tools.py**: Implementation of various tools (Python runner, file system, web search).
- **llm.py**: Abstraction layer for Language Model providers.
- **models.py**: Pydantic models defining the data structures (Task, DAG, etc.).
- **checkpoint.py**: Mechanisms for saving and loading execution state.

## Logs and Output

Execution logs are saved to `chatbot.log` in the active execution directory. This includes detailed information about task status, tool outputs, and any errors encountered during processing. Use these logs for debugging or understanding the agent's decision-making process.

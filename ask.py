#!/usr/bin/env python3
"""
Simple script to ask questions to the fault-tolerant chatbot.

Usage:
    python ask.py "Your question here"
    python ask.py  # Interactive mode
"""
import asyncio
import json
import os
import logging
import sys
import textwrap
import shutil
from pathlib import Path
from datetime import datetime

# Determine project root
PROJECT_ROOT = Path(__file__).parent.resolve()

# Create execution directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
execution_dir = PROJECT_ROOT / "executions" / f"run_{timestamp}"
execution_dir.mkdir(parents=True, exist_ok=True)

# Change working directory to execution directory
os.chdir(execution_dir)

# Configure logging with different levels for file and terminal
file_handler = logging.FileHandler("chatbot.log")
file_handler.setLevel(logging.INFO)  # Verbose file logs
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.WARNING)  # Only important messages to terminal
console_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

logging.basicConfig(
    level=logging.INFO,  # Capture everything at INFO level
    handlers=[file_handler, console_handler]
)
logger = logging.getLogger(__name__)

# Load .env file from project root
env_file = PROJECT_ROOT / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())

# Add project root to sys.path so imports work
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Now import project modules
from src.chatbot import ChatbotAPI


STATUS_ICONS = {
    "completed": "\u2713",
    "failed": "\u2717",
    "skipped": "\u2298",
    "pending": "\u25cb",
    "running": "\u25d4",
}


def truncate(text, max_len=200):
    """Truncate text with ellipsis."""
    if not text:
        return ""
    text = text.replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def print_separator(char="-", width=70):
    print(char * width)


def print_header(title, char="=", width=70):
    print()
    print(char * width)
    print(f" {title}")
    print(char * width)


def print_task(task, indent=0):
    """Print detailed info for a single task."""
    prefix = "  " * indent
    icon = STATUS_ICONS.get(task["status"], "?")
    status = task["status"].upper()

    print(f"{prefix}{icon} [{task['id']}] {task['name'] or '(unnamed)'}")
    print(f"{prefix}  Status: {status}", end="")
    if task["retry_count"] > 0:
        print(f"  (retries: {task['retry_count']})", end="")
    print()

    if task["depends_on"]:
        print(f"{prefix}  Depends on: {', '.join(task['depends_on'])}")

    if task["parent_id"]:
        print(f"{prefix}  Parent: {task['parent_id']}")

    if task["prompt"]:
        wrapped = textwrap.fill(
            task["prompt"], width=66 - indent * 2,
            initial_indent=f"{prefix}  Prompt: ",
            subsequent_indent=f"{prefix}          ",
        )
        print(wrapped)

    # Tool calls
    if task["tool_calls"]:
        print(f"{prefix}  Tool Calls ({len(task['tool_calls'])}):")
        for tc in task["tool_calls"]:
            success = tc["result"].get("success", False) if tc["result"] else None
            tc_icon = "\u2713" if success else ("\u2717" if success is False else "?")
            print(f"{prefix}    {tc_icon} {tc['name']}()")

            # Show arguments (compact)
            args_str = json.dumps(tc["arguments"], indent=None)
            if len(args_str) > 120:
                args_str = args_str[:120] + "..."
    if task["error"]:
        print(f"{prefix}  Error: {task['error']}")

    print()


async def ask(question: str):
    api = ChatbotAPI()

    print_header(f"Question: {question}")
    print("Processing (decomposing into sub-tasks)...\n")

    response = await api.ask(question, include_details=True)

    details = response.task_details

    # DAG overview
    print_header("DAG Overview")
    print(f"  DAG ID: {details['dag_id']}")
    print(f"  Question: {details['user_question']}")
    stats = details["stats"]
    print(f"  Tasks: {stats['total']} total | "
          f"{stats['completed']} completed | "
          f"{stats['failed']} failed | "
          f"{stats['skipped']} skipped")
    print(f"  Completion: {stats['completion_rate']:.0%}")
    if details["dead_letter_queue"] > 0:
        print(f"  Dead Letter Queue: {details['dead_letter_queue']} entries")

    # Task details
    print_header("Task Execution Details")
    for task in details["tasks"]:
        print_task(task)

    # Final answer or help request
    if response.needs_user_help:
        print_header("Help Needed", char="!")
        print(response.help_request)
        print()
        print_separator("!")
    else:
        print_header("Final Answer")
        print(response.answer)
        print_separator("=")

    return response


def main():
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        print("Fault-Tolerant Chatbot")
        print("Enter your question (or 'quit' to exit):\n")
        question = input("> ").strip()
        if question.lower() in ('quit', 'exit', 'q'):
            return

    asyncio.run(ask(question))


if __name__ == "__main__":
    main()

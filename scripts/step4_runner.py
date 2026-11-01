import json
from pathlib import Path
from typing import Any

from anthropic import Anthropic
from anthropic.types import Message
from dotenv import load_dotenv

from agent.claude_raw import tool_definitions
from agent.loop import run

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts" / "step3"
RUN_FILE = ARTIFACTS / "run.json"

if __name__ == "__main__":
    load_dotenv()
    client = Anthropic()

    print("hello")
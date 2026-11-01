import json
from pathlib import Path
from typing import Any

from anthropic import Anthropic
from anthropic.types import Message
from dotenv import load_dotenv

from agent.claude_raw import tool_definitions
from agent.loop import Context, run

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts" / "step4"
RUN_FILE = ARTIFACTS / "run.json"
AGENT_FILE = ARTIFACTS / "agent.log"

if __name__ == "__main__":
    load_dotenv()
    client = Anthropic()

    prompt = (
        "The PM wants each product's stock shown in search results. Find where search is implemented in shop/ and make the change."
    )

    def messages_wrapper(messages: list[dict]) -> Message:
        return client.messages.create(
            model="claude-sonnet-5-5",
            max_tokens=10240,
            tools=tool_definitions(),
            messages=messages,
        )

    output = run(prompt, messages_wrapper, ctx=Context("investigator"))

    # Create all parent directories if they don't exist
    RUN_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(RUN_FILE, "w") as f:
        json.dump(output, f, indent=2)

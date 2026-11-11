import json
from pathlib import Path

from anthropic import Anthropic
from anthropic.types import Message
from dotenv import load_dotenv

from agent.mcp_client import run_with_mcp
from agent.mcp_server import build_server

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts" / "step5"
RUN_FILE = ARTIFACTS / "run.json"

if __name__ == "__main__":
    load_dotenv()
    client = Anthropic()

    prompt = (
        "The PM wants each product's stock shown in search results. Find where search is implemented in shop/ and make the change."
    )

    def messages_wrapper(messages: list[dict], tools) -> Message:
        return client.messages.create(
            model="claude-sonnet-5-5",
            max_tokens=10240,
            tools=tools,
            messages=messages,
        )

    server = build_server("investigator")

    output = run_with_mcp(prompt, messages_wrapper, server)

    # Create all parent directories if they don't exist
    RUN_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(RUN_FILE, "w") as f:
        json.dump(output, f, indent=2)

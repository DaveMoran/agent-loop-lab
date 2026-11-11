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

    prompt = """In shop/app.py, search_products() runs "SELECT * FROM products WHERE name LIKE ?".
        Change that query to name its columns explicitly — id, name, price, stock — and update
        the functions docstring to say that results include each products current stock.
        Apply the edit to shop/app.py."""

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

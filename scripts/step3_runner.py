import json
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

from agent.claude_raw import tool_definitions
from agent.loop import run

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts" / "step3"
RUN_FILE = ARTIFACTS / "run.json"

if __name__ == "__main__":
    load_dotenv()
    client = Anthropic()

    prompt = (
        "Support says some checkouts are failing. "
        "Find out why, and which orders are affected."
    )

    def messages_wrapper(messages):
        return client.messages.create(
            model="claude-sonnet-5-5",
            max_tokens=10240,
            tools=tool_definitions(),
            messages=messages,
        )

    output = run(prompt, messages_wrapper)

    # Create all parent directories if they don't exist
    RUN_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(RUN_FILE, "w") as f:
        json.dump(output, f, indent=2)

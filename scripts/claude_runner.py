import json
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

from agent.claude_raw import extract_tool_calls, tool_definition, tool_result
from agent.tools import search_logs

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts" / "step2"
ANTHROPIC_FILE = ARTIFACTS / "anthropic_tool_request.json"
ANTHROPIC_SECOND = ARTIFACTS / "anthropic_second_request.json"
ANTHROPIC_FINAL = ARTIFACTS / "anthropic_final_response.json"

if __name__ == "__main__":
    load_dotenv()
    client = Anthropic()

    messages = [{"role": "user", "content": "Give me all logs that are warnings"}]

    response = client.messages.create(
        model="claude-opus-5-5",
        max_tokens=1024,
        tools=[tool_definition()],
        messages=messages,
    )

    with open(ANTHROPIC_FILE, "w") as f:
        f.write(response.model_dump_json(indent=2))

    messages += ({"role": "assistant", "content": response.model_dump()["content"]},)
    tool_calls = extract_tool_calls(response.model_dump())

    for call in tool_calls:
        result = tool_result(call["id"], search_logs(**call["arguments"]))
        messages.append(result)

    with open(ANTHROPIC_SECOND, "w") as f:
        clean_input = [
            item.model_dump() if hasattr(item, "model_dump") else item
            for item in messages
        ]

        f.write(json.dumps({"messages": clean_input}))

    second_response = client.messages.create(
        model="claude-opus-5-5",
        max_tokens=1024,
        tools=[tool_definition()],
        messages=messages,
    )

    with open(ANTHROPIC_FINAL, "w") as f:
        f.write(second_response.model_dump_json(indent=2))

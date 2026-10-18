import json
from typing import Any

from agent.tools import SEARCH_LOGS_PARAMETERS


# Define list of tools
def tool_definition() -> dict:
    return {
        "type": "function",
        "name": "search_logs",
        "description": "Search through the logs of the application",
        "parameters": SEARCH_LOGS_PARAMETERS,
    }


def extract_tool_calls(response: dict) -> list[dict]:
    tool_calls = []
    for output in response["output"]:
        if output["type"] == "function_call":
            tool_calls.append(
                {
                    "id": output["call_id"],
                    "name": output["name"],
                    "arguments": json.loads(output["arguments"]),
                }
            )

    return tool_calls


def tool_result(call_id: str, result: Any) -> dict:
    return {
        "type": "function_call_output",
        "call_id": call_id,
        "output": json.dumps(result),
    }

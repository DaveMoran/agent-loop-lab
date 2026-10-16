import json

from agent.tools import SEARCH_LOGS_PARAMETERS

# Define list of tools
def tool_definition() -> dict:
    return {
        "name": "search_logs",
        "description": "Search through the logs of the application",
        "input_schema": SEARCH_LOGS_PARAMETERS
    }

def extract_tool_calls(response: dict) -> list[dict]:
    tool_calls = []
    for output in response["content"]:
        if output["type"] == "tool_use":
            tool_calls.append({
                "id": output["id"],
                "name": output["name"],
                "arguments": output["input"]
            })

    return tool_calls

def tool_result(call_id: str, result) -> dict:
    return {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": call_id,
                "content": json.dumps(result)
            }
        ]
    }

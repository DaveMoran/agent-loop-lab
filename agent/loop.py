import json
import os
from pathlib import Path

from collections.abc import Callable

from agent.claude_raw import extract_tool_calls, tool_result
from agent.tools import TOOLS
from agent.permissions import check


def run(
    prompt: str,
    call_model: Callable,
    tools: dict = TOOLS,
    max_turns: int = 10,
    ctx=None,
) -> dict:
    stopped = "answer"
    messages = [{"role": "user", "content": prompt}]
    turns = 0

    while turns < max_turns:
        model_response = call_model(messages)
        turns += 1
        answer = ""

        if hasattr(model_response, "model_dump"):
            model_response = model_response.model_dump()

        tool_calls = extract_tool_calls(model_response)
        messages.append({"role": "assistant", "content": model_response["content"]})
        temp_calls = []
        for call in tool_calls:
            denial = check(ctx, call["name"], call["arguments"])
            if denial:
                audit_log_file = Path(os.getenv("AGENT_LOG", "logs/agent.log"))
                audit_log_dir = audit_log_file.parent

                audit_log_dir.mkdir(parents=True, exist_ok=True)
                audit_log_file.write_text(json.dumps(denial), newline="\n")
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": call["id"],
                                "is_error": True,
                                "content": json.dumps(denial),
                            }
                        ],
                    }
                )
            else:
                try:
                    result = tool_result(
                        call["id"], tools[call["name"]](**call["arguments"])
                    )
                    temp_calls.append(result)
                except Exception as e:
                    # Code that runs for any other unexpected error
                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "is_error": True,
                                    "content": str(e),
                                }
                            ],
                        }
                    )

        if model_response["stop_reason"] == "tool_use":
            if len(temp_calls) > 1:
                message = {
                    "role": "user",
                    "content": [],
                }

                for call in temp_calls:
                    message["content"].append(call["content"][0])

                messages.append(message)
            elif len(temp_calls) == 1:
                messages.append(temp_calls[0])
        else:
            answer = model_response["content"][-1]["text"]
            break

    if turns == max_turns:
        answer = "Max turns reached (" + str(max_turns) + ")"
        stopped = "max_turns"

    return {
        "answer": answer,
        "messages": messages,
        "turns": turns,
        "stopped": stopped,
    }

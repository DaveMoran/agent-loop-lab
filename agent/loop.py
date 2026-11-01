import json
from collections.abc import Callable

from agent.audit import write_to_audit_log
from agent.claude_raw import extract_tool_calls, tool_result
from agent.tools import TOOLS
from agent.permissions import Context, check


def run(
    prompt: str,
    call_model: Callable,
    tools: dict = TOOLS,
    max_turns: int = 10,
    ctx: Context = Context(),
) -> dict:
    stopped = "max_turns"
    answer = "Max turns reached (" + str(max_turns) + ")"
    messages = [{"role": "user", "content": prompt}]
    turns = 0

    while turns < max_turns:
        model_response = call_model(messages)
        turns += 1

        if hasattr(model_response, "model_dump"):
            model_response = model_response.model_dump()

        tool_calls = extract_tool_calls(model_response)
        messages.append({"role": "assistant", "content": model_response["content"]})
        temp_calls = []
        for call in tool_calls:
            try:
                denial = check(ctx, call["name"], call["arguments"])
                if denial:
                    write_to_audit_log(
                        {
                            "event": "permission_denied",
                            **denial,
                            "arguments": call["arguments"],
                        }
                    )

                    temp_calls.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": call["id"],
                            "is_error": True,
                            "content": json.dumps(denial),
                        }
                    )
                else:
                    result = tool_result(
                        call["id"], tools[call["name"]](**call["arguments"])
                    )
                    temp_calls.append(result["content"][0])
            except Exception as e:
                temp_calls.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": call["id"],
                        "is_error": True,
                        "content": str(e),
                    }
                )

        if model_response["stop_reason"] == "tool_use":
            message = {
                "role": "user",
                "content": [],
            }
            for call in temp_calls:
                message["content"].append(call)

            messages.append(message)
        else:
            answer = model_response["content"][-1]["text"]
            stopped = "answer"
            break

    return {
        "answer": answer,
        "messages": messages,
        "turns": turns,
        "stopped": stopped,
    }

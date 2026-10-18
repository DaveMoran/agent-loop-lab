from agent.claude_raw import extract_tool_calls, tool_result
from agent.tools import TOOLS

def run(
    prompt: str, call_model, tools: dict | None = TOOLS, max_turns: int = 10
) -> dict:
    stopped = "answer"
    messages = [{"role": "user", "content": prompt}]
    turns = 0

    while turns < max_turns:
        model_response = call_model(messages)
        turns += 1

        tool_calls = extract_tool_calls(model_response)
        messages.append({"role": "assistant", "content": model_response["content"]})
        for call in tool_calls:
            try:
                result = tool_result(
                    call["id"], tools[call["name"]](**call["arguments"])
                )
                messages.append(result)
            except Exception as e:
                # Code that runs for any other unexpected error
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "tool_result", "is_error": True, "content": str(e)}
                        ],
                    }
                )

        if model_response["stop_reason"] != "tool_use":
            messages.append({"role": "user", "content": model_response["content"][0]})
            break

    if turns == max_turns:
        messages.append(
            {
                "role": "user",
                "content": {
                    "type": "tool_result",
                    "is_error": True,
                    "text": f"Max turns reached ({max_turns})",
                },
            }
        )
        stopped = "max_turns"

    return {
        "answer": messages[len(messages) - 1]["content"]["text"],
        "messages": messages,
        "turns": turns,
        "stopped": stopped,
    }

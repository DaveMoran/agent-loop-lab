from agent.claude_raw import extract_tool_calls, tool_result


def run(
    prompt: str, call_model, tools: dict | None = None, max_turns: int = 10
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
            result = tool_result(call["id"], tools[call["name"]](**call["arguments"]))
            messages.append(result)

        if model_response["stop_reason"] != "tool_use":
            messages.append({"role": "user", "content": model_response["content"][0]})
            break

    return {
        "answer": messages[len(messages) - 1]["content"]["text"],
        "messages": messages,
        "turns": turns,
        "stopped": stopped,
    }

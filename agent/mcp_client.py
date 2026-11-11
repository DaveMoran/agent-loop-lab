from anyio.from_thread import start_blocking_portal
from mcp import Client

from agent.claude_raw import extract_tool_calls


def tool_definitions_from_mcp(list_tools_result) -> list[dict]:
    tool_defs = []

    for tool in list_tools_result.tools:
        tool_meta = {
            "name": tool.name,
            "description": tool.description, 
            "input_schema": tool.input_schema
        }

        tool_defs.append(tool_meta)
    return tool_defs


def to_tool_result(tool_use_id: str, call_tool_result) -> dict:    
    return {
        "type": "tool_result", 
        "tool_use_id": tool_use_id,
        "content": "\n".join(getattr(c, "text", "") for c in call_tool_result.content), 
        "is_error": call_tool_result.is_error
    }


def run_with_mcp(prompt, call_model, server, max_turns=10) -> dict:
    stopped = "max_turns"
    answer = "Max turns reached (" + str(max_turns) + ")"
    messages = [{"role": "user", "content": prompt}]
    turns = 0

    with start_blocking_portal() as portal:
        with portal.wrap_async_context_manager(Client(server)) as client:
            listed = portal.call(client.list_tools)
            tools = tool_definitions_from_mcp(listed)

            def execute(name, arguments):
                # Client(raise_exceptions=False) is the default, so per-call
                # failures - an unknown tool, a permission denial, a bad path -
                # come back as a CallToolResult with is_error set, and go to the
                # model as a tool_result it can act on.
                #
                # Protocol errors (MCPError, which mcp/server/.../tools/base.py
                # re-raises rather than wrapping) and a dead transport RAISE
                # instead. Those are session-level: the next call fails the same
                # way, so turning them into a tool_result would only tell the
                # model to retry something impossible until max_turns. They are
                # left to propagate out of run_with_mcp deliberately.
                return portal.call(client.call_tool, name, arguments)
        
            while turns < max_turns:
                model_response = call_model(messages, tools)
                turns += 1
        
                if hasattr(model_response, "model_dump"):
                    model_response = model_response.model_dump()
        
                tool_calls = extract_tool_calls(model_response)
                messages.append({"role": "assistant", "content": model_response["content"]})
                temp_calls = []

                for call in tool_calls:
                    result = execute(call["name"], call["arguments"])
                    tool_result = to_tool_result(call["id"], result)
                    temp_calls.append(tool_result)

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
        
"""to_tool_result(tool_use_id: str, call_tool_result) -> dict
        - Turns the SDK's CallToolResult into one Claude tool_result block:
          {"type": "tool_result", "tool_use_id": ..., "content": <str>, "is_error": <bool>}
          where content holds the result's text. is_error mirrors the server's isError.
    run_with_mcp(prompt, call_model, server, max_turns=10) -> dict
        - The Step 4 loop, with the tools living behind an MCP server. `server` is
          anything mcp.Client accepts (an MCPServer object in these tests).
        - call_model is called as call_model(messages, tools), where tools is the list
          from tool_definitions_from_mcp. The client discovers the tools; you don't
          hand-write them a second time.
        - Same return shape as loop.run: {"answer", "messages", "turns", "stopped"},
          "stopped" is "answer" or "max_turns", all results of a turn go back in one
          user message in call order.
        - It has no ctx: the level is the server's. It must not add a second gate with a
          level of its own (an operator server's write goes through), and a denial is
          logged once, by the server.
        - It's a plain (synchronous) function; how you bridge to the async SDK is up to you."""


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
        "content": call_tool_result.content[0].text, 
        "is_error": call_tool_result.is_error
    }


def run_with_mcp(prompt, call_model, server, max_turns=10) -> dict:
    return {}
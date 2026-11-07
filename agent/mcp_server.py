from typing import Annotated

from mcp.types import ToolAnnotations
from pydantic import Field

from mcp.server import MCPServer
import agent.tools as tools


def build_server(level: str = "investigator") -> MCPServer:
    if level.lower() not in ["operator", "investigator"]:
        raise ValueError(f"Level {level} is not supported for this tool")

    mcp = MCPServer(level)

    @mcp.tool(
        title="Search the logs",
        description="Search the logs by a specific query and level of log",
        annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False),
    )
    def search_logs(
        query: Annotated[str, Field(description="Title or author to search for.")],
        level: Annotated[str, Field(description="Level of severity of the log.")],
        limit: Annotated[
            int, Field(ge=1, description="Maximum number of results.")
        ] = 20,
    ):
        """Search the logs by a specific query and level of log"""
        return tools.search_logs(query, level, limit)

    @mcp.tool(
        title="Read a file",
        description="Reads the content of a file that is located in the shop directory",
        annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False),
    )
    def read_file(path: str):
        return tools.read_file(path)

    @mcp.tool(
        title="Write to a file",
        description="Write the provided content to a new or existing file",
        annotations=ToolAnnotations(read_only_hint=False, open_world_hint=False),
    )
    def write_file(path: str, content: str):
        return tools.write_file(path, content)

    @mcp.tool(
        title="Hand off the work back to the human",
        description="Due to an error or block, a handoff will be created for the user",
        annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False),
    )
    def create_handoff(summary: str, evidence: list[str], requested_action: str):
        return tools.create_handoff(summary, evidence, requested_action)

    return mcp


mcp = build_server()


if __name__ == "__main__":
    mcp.run()

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
        level: Annotated[str | None, Field(description="Level of severity of the log.")] = None,
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
    def read_file(
        path: Annotated[str, Field(description="Location of the file to be read")],
    ):
        return tools.read_file(path)

    @mcp.tool(
        title="Write to a file",
        description="Write the provided content to a new or existing file",
        annotations=ToolAnnotations(read_only_hint=False, open_world_hint=False),
    )
    def write_file(
        path: Annotated[
            str, Field(description="Location of the file to be written to")
        ],
        content: Annotated[
            str, Field(description="Content that will be written to the file")
        ],
    ):
        return tools.write_file(path, content)

    @mcp.tool(
        title="Hand off the work back to the human",
        description="Due to an error or block, a handoff will be created for the user",
        annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False),
    )
    def create_handoff(
        summary: Annotated[
            str,
            Field(
                description="A note to be sent to the human regarding why this is being handed off"
            ),
        ],
        evidence: Annotated[
            list[str],
            Field(
                description="A list of strings that add context to the handoff and support the summary"
            ),
        ],
        requested_action: Annotated[
            str,
            Field(
                description="A suggested next step for the human to take to continue"
            ),
        ],
    ):
        return tools.create_handoff(summary, evidence, requested_action)

    return mcp


mcp = build_server()


if __name__ == "__main__":
    mcp.run()

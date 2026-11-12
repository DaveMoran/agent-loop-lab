import json
import os
from typing import Annotated, Literal

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

import agent.tools as tools
from agent.audit import write_to_audit_log
from agent.permissions import Context, check


def build_server(level: str = "investigator") -> MCPServer:
    if level.lower() not in ["operator", "investigator"]:
        raise ValueError(f"Level {level} is not supported for this tool")

    level = level.lower()
    mcp = MCPServer(level)

    def gate(name: str, arguments: dict) -> None:
        """Run one call through check() before the tool body does any work.

        Every tool starts with this, so the gate is one decision in one place the
        way loop.run's single check() was in Step 4. Tools that need no level
        today still call it: the next tool added is gated by default, not by
        whoever remembers to add it.

        check() returns None to allow, or a denial dict. The audit line keeps the
        arguments; the denial the model sees never does.
        """
        denial = check(Context(level), name, arguments)
        if denial is None:
            return

        write_to_audit_log(
            {"event": "permission_denied", **denial, "arguments": arguments}
        )
        # ToolError is an anticipated failure, so the payload reaches the model:
        # check() already chose the reason and whether to suggest a handoff.
        raise ToolError(json.dumps(denial))

    @mcp.tool(
        title="Search the logs",
        description="Search the logs by a specific query and level of log",
        annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False),
    )
    def search_logs(
        query: Annotated[
            str,
            Field(
                description="Text to look for anywhere in a log line, matched "
                "case-insensitively. An event name like 'payment_timeout', a "
                "product term, or a reason like 'out_of_stock'."
            ),
        ],
        level: Annotated[
            Literal["INFO", "ERROR", "WARN"] | None,
            Field(description="Only return lines at this level. Omit for all levels."),
        ] = None,
        limit: Annotated[
            int, Field(ge=1, description="Maximum number of results.")
        ] = 20,
    ):
        """Search the logs by a specific query and level of log"""
        gate("search_logs", {"query": query, "level": level, "limit": limit})
        return tools.search_logs(query, level, limit)

    @mcp.tool(
        title="Read a file",
        description="Reads the content of a file that is located in the shop directory",
        annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False),
    )
    def read_file(
        path: Annotated[str, Field(description="Location of the file to be read")],
    ):
        gate("read_file", {"path": path})
        try:
            return tools.read_file(path)
        except FileNotFoundError:
            raise ToolError(f"File {path} not found")
        except OSError as exc:
            # Anything else the filesystem refuses - a directory, a bad encoding -
            # is a mistake the model can correct, so it needs the real message.
            raise ToolError(f"Cannot read {path}: {exc}")

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
        gate("write_file", {"path": path, "content": content})
        try:
            return tools.write_file(path, content)
        except FileNotFoundError as exc:
            raise ToolError(str(exc))
        except OSError as exc:
            # gate() has already refused anything outside shop/, so the
            # PermissionError tools.write_file raises for scope can't land here.
            raise ToolError(f"Cannot write {path}: {exc}")

    @mcp.tool(
        title="Hand off the work back to the human",
        description="Due to an error or block, a handoff will be created for the user",
        # Not read-only: this appends a handoff record to the audit log.
        annotations=ToolAnnotations(read_only_hint=False, open_world_hint=False),
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
        gate(
            "create_handoff",
            {
                "summary": summary,
                "evidence": evidence,
                "requested_action": requested_action,
            },
        )
        return tools.create_handoff(summary, evidence, requested_action)

    return mcp


if __name__ == "__main__":
    # Built here, not at import time: a bad AGENT_LEVEL should fail the process
    # that serves, not every module that imports this one.
    build_server(os.environ.get("AGENT_LEVEL", "investigator")).run()

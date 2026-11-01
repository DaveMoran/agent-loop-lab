from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timezone

from agent.tools import is_inside_shop


@dataclass(frozen=True)
class Context:
    level: str = "investigator"

    def __post_init__(self):
        if self.level.lower() not in ["investigator", "operator"]:
            raise ValueError(
                f"Invalid level. Expected valid string value, got {type(self.level)}"
            )


def check(ctx: Context, name: str, arguments: dict) -> dict | None:
    is_read_only = True
    if hasattr(ctx, "level") and ctx.level == "operator":
        is_read_only = False

    user_level = ctx.level if hasattr(ctx, "level") else "None"

    if name == "write_file":
        if is_read_only:
            return {
                "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                "error": "permission_denied",
                "event": "permission_denied",
                "tool": name,
                "level": user_level,
                "reason": f"{user_level} does not have permission to run tool f{name}",
                "suggestion": "Use tool create_handoff",
                "arguments": arguments,
            }
    if "path" in arguments:
        if is_inside_shop(arguments["path"]):
            return None
        else:
            return {
                "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                "error": "permission_denied",
                "event": "permission_denied",
                "tool": name,
                "level": user_level,
                "reason": f"Access to file {arguments["path"]} not allowed.",
                "suggestion": "Use tool create_handoff",
                "arguments": arguments,
            }

    return None

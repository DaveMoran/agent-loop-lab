from dataclasses import dataclass
from enum import IntEnum
from datetime import datetime, timezone

from agent.tools import is_inside_shop


class Permission(IntEnum):
    INVESTIGATOR = 1
    OPERATOR = 2


REQUIRED_LEVEL: dict[str, Permission] = {
    "write_file": Permission.OPERATOR,
}


@dataclass(frozen=True)
class Context:
    level: str = "investigator"

    def __post_init__(self):
        if self.level.upper() not in Permission.__members__.keys():
            raise ValueError(
                f"Invalid level. Expected valid string value, got {self.level}"
            )

    @property
    def permission_level(self) -> Permission:
        return Permission[self.level.upper()]


def check(ctx: Context, name: str, arguments: dict) -> dict | None:
    user_level = ctx.level

    if ctx.permission_level < REQUIRED_LEVEL.get(name, Permission.INVESTIGATOR):
        return {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "error": "permission_denied",
            "event": "permission_denied",
            "tool": name,
            "level": user_level,
            "reason": f"{user_level} ({ctx.permission_level}) does not have permission to run tool {name}.",
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

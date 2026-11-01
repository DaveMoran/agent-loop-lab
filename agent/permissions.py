from dataclasses import dataclass
from enum import IntEnum

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
        if self.level.upper() not in Permission.__members__:
            raise ValueError(
                f"Invalid level. Expected valid string value, got {self.level}"
            )

    @property
    def permission_level(self) -> Permission:
        return Permission[self.level.upper()]


def build_denial(tool: str, level: str, args: dict, reason: str) -> dict:
    return {
        "error": "permission_denied",
        "event": "permission_denied",
        "tool": tool,
        "level": level,
        "reason": reason,
        "suggestion": "Use tool create_handoff",
        "arguments": args,
    }


def check(ctx: Context, name: str, arguments: dict) -> dict | None:
    user_level = ctx.level

    if ctx.permission_level < REQUIRED_LEVEL.get(name, Permission.INVESTIGATOR):
        return build_denial(
            name,
            user_level,
            arguments,
            f"{user_level} ({ctx.permission_level}) does not have permission to run tool {name}.",
        )

    if "path" in arguments:
        path = arguments["path"]
        try:
            allowed = isinstance(path, str) and is_inside_shop(path)
        except ValueError:
            allowed = False

        if allowed:
            return None
        else:
            return build_denial(
                name,
                user_level,
                arguments,
                f"Access to file {path!r} not allowed: path must be a string inside shop/.",
            )

    return None

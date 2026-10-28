from dataclasses import dataclass
from enum import Enum

from agent.tools import read_file

@dataclass(frozen=True)
class Context:
    level: str = "investigator"

    def __post_init__(self):
        if self.level.lower() not in ['investigator', 'operator']:
            raise ValueError(
                f"Invalid level. Expected valid string value, got {type(self.level)}"
            )

def check(ctx: Context, name: str, arguments: dict) -> dict | None:
    if name == 'write_file':
        if ctx.level == 'investigator':
            return {
                "error": "permission_denied", 
                "tool": name, 
                "level": ctx.level,
                "reason": f"{ctx.level} does not have permission to run tool f{name}", 
                "suggestion": "Use tool create_handoff"
            }

    if "path" in arguments:
        try:
            file = read_file(arguments["path"])
        except PermissionError as e:
            return {
                "error": "permission_denied", 
                "tool": name, 
                "level": ctx.level,
                "reason": f"Access to file {arguments["path"]} not allowed.", 
                "suggestion": "Use tool create_handoff"
            }


    
    return None
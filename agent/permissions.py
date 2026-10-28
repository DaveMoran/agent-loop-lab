from dataclasses import dataclass
from enum import Enum

@dataclass(frozen=True)
class Context:
    level: str = "investigator"

    def __post_init__(self):
        if self.level.lower() not in ['investigator', 'operator']:
            raise ValueError(
                f"Invalid level. Expected valid string value, got {type(self.level)}"
            )

def check(ctx: Context, name: str, arguments: dict) -> dict | None:
    return None
"""JSON-lines logger. One record per event, written to logs/app.log (or SHOP_LOG)."""
import json
import os
import time
from pathlib import Path


def log_path() -> Path:
    return Path(os.environ.get("SHOP_LOG", "logs/app.log"))


def emit(level: str, event: str, **fields) -> None:
    path = log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"ts": round(time.time(), 3), "level": level, "event": event, **fields}
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")

import json
import os
from datetime import datetime, timezone

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"


def write_to_audit_log(entry: dict) -> None:
    audit_log_file = Path(os.getenv("AGENT_LOG", LOG_DIR / "agent.log"))
    audit_log_dir = audit_log_file.parent

    audit_log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds")

    with open(audit_log_file, "a", encoding="utf-8") as file:
        file.write(json.dumps({"ts": timestamp, **entry}) + "\n")

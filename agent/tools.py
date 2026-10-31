"""
Individual tools that the agent will call that will allow it to go back and forth
with the client while
"""

import json
import os
import sqlite3
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from shop.db import db_path
from shop.logs import log_path

LOGS = Path(__file__).resolve().parents[1] / "logs"
LOG_FILE = LOGS / "agent.json"

SEARCH_LOGS_PARAMETERS: dict = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "The string to check against each line in the logs",
        },
        "level": {
            "type": "string",
            "enum": ["INFO", "WARN", "ERROR"],
            "description": "The level of the log to filter for",
        },
        "limit": {
            "type": "integer",
            "description": "The maximum number of logs to return",
        },
    },
    "required": ["query"],
    "additionalProperties": False,
}

READ_FILE_PARAMETERS: dict = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "Path to file that will be read"}
    },
    "required": ["path"],
    "additionalProperties": False,
}

RUN_QUERY_PARAMETERS: dict = {
    "type": "object",
    "properties": {
        "sql": {
            "type": "string",
            "description": "SQL command to be executed against db",
        }
    },
    "required": ["sql"],
    "additionalProperties": False,
}

WRITE_FILE_PARAMETERS: dict = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "Path to file that will be read"},
        "content": {
            "type": "string",
            "description": "Content that will be written to the file",
        },
    },
    "required": ["path", "content"],
    "additionalProperties": False,
}

HANDOFF_PARAMETERS: dict = {
    "type": "object",
    "properties": {
        "summary": {
            "type": "string",
            "description": "instructions needed for the handoff to be performed successfully",
        },
        "evidence": {
            "type": "array",
            "description": "A list of strings that will be used to providde additional context for the Agent",
            "items": {
                "type": "string",
            },
        },
        "requested_action": {
            "type": "string",
            "description": "Action/Tool requested by the original role",
        },
    },
    "required": ["summary", "evidence", "requested_action"],
    "additionalProperties": False,
}


def search_logs(
    query: str | None, level: str | None = None, limit: int = 20
) -> list[dict]:
    # Immediate return for 0 limit
    if limit < 1:
        return []

    with open(log_path()) as file:
        final_logs = []
        # Step 1: Filter out all queries
        for line in file:
            is_match = False
            try:
                line_as_json = json.loads(line)
            except json.JSONDecodeError:
                continue

            # If no query sent, send max results
            if not query and not level:
                final_logs.append(line_as_json)
                continue
            else:
                # If query AND level sent
                if query and level:
                    is_match = matches_query(query, line) and matches_level(
                        level, line_as_json
                    )
                else:
                    # if query
                    if query and matches_query(query, line):
                        is_match = True

                    # if level
                    if level:
                        if matches_level(level, line_as_json):
                            is_match = True

            if is_match:
                final_logs.append(line_as_json)

        return final_logs[-limit:]


def matches_query(query: str, line: str) -> bool:
    return query.casefold() in line.casefold()


def matches_level(level: str, json: dict) -> bool:
    return level.casefold() == json["level"].casefold()


def read_file(path: str) -> str:
    # Check path to ensure that file is inside shop directory (PermissionError)
    ROOT = Path(__file__).resolve().parents[1]
    SHOP_DIR = ROOT / "shop"

    resolved_path = Path(path).resolve()
    try:
        if path.startswith("shop/") and not path.startswith("shop/.."):
            with open(ROOT / path) as file:
                content = file.read()
                return content
        else:
            if resolved_path.is_relative_to(SHOP_DIR):
                with open(path) as file:
                    content = file.read()
                    return content
            else:
                raise PermissionError("Error: File is not in supported directory")
    except FileNotFoundError:
        # If file is missing, raise FileNotFoundError
        raise FileNotFoundError("Error: The file does not exist.")


def open_readonly_db() -> sqlite3.Connection:
    # Double-check your file path actually exists
    if not os.path.exists(db_path()):
        raise FileNotFoundError(
            f"Database file not found at: {os.path.abspath(db_path())}"
        )

    # Convert file path to a URI
    db_uri = db_path().resolve().as_uri()

    try:
        # mode=rw throws an error if the file does not exist
        conn = sqlite3.connect(f"{db_uri}?mode=ro", uri=True)
        return conn
    except sqlite3.OperationalError:
        print("Database file was not found!")
        raise


def run_query(sql: str) -> list[dict]:
    formatted_sql = sql.strip().casefold()
    if not formatted_sql.startswith("select"):
        raise PermissionError("This tool only allows for SELECT queries")

    conn = open_readonly_db()
    conn.row_factory = sqlite3.Row  # <-- This is the key change
    cursor = conn.cursor()

    cursor.execute(formatted_sql)
    row = cursor.fetchone()

    dict_rows = []

    while row is not None:
        dict_rows.append(dict(row))
        row = cursor.fetchone()

    return dict_rows


def write_file(path: str, content: str) -> dict:
    # Check path to ensure that file is inside shop directory (PermissionError)
    ROOT = Path(__file__).resolve().parents[1]
    SHOP_DIR = ROOT / "shop"

    resolved_path = Path(path).resolve()
    try:
        if path.startswith("shop/") and not path.startswith("shop/.."):
            with open(ROOT / path, "w") as file:
                file.write(content)
                return {}
        else:
            if resolved_path.is_relative_to(SHOP_DIR):
                with open(path, "w") as file:
                    file.write(content)
                    return {}
            else:
                raise PermissionError("Error: File is not in supported directory")
    except FileNotFoundError:
        # If file is missing, raise FileNotFoundError
        raise FileNotFoundError("Error: The file does not exist.")


def create_handoff(summary: str, evidence: list[str], requested_action: str) -> dict:
    if not summary.strip():
        raise ValueError("Summary can not be empty")
    if not requested_action.strip():
        raise ValueError("Requested Action can not be empty")
    if len(evidence) < 1 or not isinstance(evidence, list):
        raise ValueError("Evidence can not be empty")

    entry = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "id": str(uuid.uuid4()),
        "summary": summary,
        "evidence": evidence,
        "requested_action": requested_action,
        "event": "handoff",
        "status": "open",
    }

    audit_log_file = Path(os.getenv("AGENT_LOG", "logs/agent.log"))
    audit_log_dir = audit_log_file.parent

    audit_log_dir.mkdir(parents=True, exist_ok=True)
    with open(audit_log_file, "a", encoding="utf-8") as file:
        file.write(json.dumps(entry) + "\n")

    return entry


TOOLS: dict[str, Callable] = {
    "search_logs": search_logs,
    "read_file": read_file,
    "run_query": run_query,
    "write_file": write_file,
    "create_handoff": create_handoff,
}

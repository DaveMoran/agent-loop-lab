"""Step 5 acceptance tests, part 2: evidence from the Inspector and one real run. No API calls.

Commit three things in artifacts/step5/:

    inspector.md          Your notes from calling each tool by hand in the MCP Inspector.
                          One section per tool you called: search_logs, read_file and
                          create_handoff at least, plus one denied call (write_file on an
                          investigator server, or a path outside shop/). For each, paste
                          the result the Inspector showed (trimmed is fine) and one line on
                          what the Inspector showed you that your loop never did.
    run.json              run_with_mcp()'s return value from one real Claude run against an
                          INVESTIGATOR server: Step 4's scenario again, a denied write_file
                          followed by create_handoff.
    agent.log             The audit log from that run (AGENT_LOG), unedited.
    where-the-check-lives.md
                          At least 150 words: where the permission check lives now (server,
                          client, or both and what each one does), and your defence of it.
                          Say what happens when a client other than your loop connects.

Suggested setup for the run:
    rm -f artifacts/step5/agent.log
    AGENT_LOG=artifacts/step5/agent.log  (set before the run)
    the same prompt you used for Step 4.

Don't edit this file; if a test looks wrong, raise it.
"""
import json
import re
from pathlib import Path

import pytest

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts" / "step5"
RUN_FILE = ARTIFACTS / "run.json"
LOG_FILE = ARTIFACTS / "agent.log"
INSPECTOR_FILE = ARTIFACTS / "inspector.md"
NOTE_FILE = ARTIFACTS / "where-the-check-lives.md"
SECRET_PATTERN = re.compile(r"sk-[A-Za-z0-9_\-]{20,}|sk-ant-[A-Za-z0-9_\-]{10,}")


@pytest.fixture(scope="module")
def run():
    assert RUN_FILE.exists(), "missing artifacts/step5/run.json: save run_with_mcp()'s return value from a real run"
    return json.loads(RUN_FILE.read_text())


@pytest.fixture(scope="module")
def log():
    assert LOG_FILE.exists(), "missing artifacts/step5/agent.log: copy the AGENT_LOG from the same run"
    return [json.loads(line) for line in LOG_FILE.read_text().splitlines() if line.strip()]


def blocks(message, kind):
    content = message["content"]
    if isinstance(content, str):
        return []
    return [b for b in content if b.get("type") == kind]


def calls_and_results(run):
    pairs = []
    messages = run["messages"]
    for i, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        uses = blocks(m, "tool_use")
        if not uses:
            continue
        results = {r["tool_use_id"]: r for r in blocks(messages[i + 1], "tool_result")}
        pairs.extend((u, results.get(u["id"])) for u in uses)
    return pairs


def content_text(result):
    content = result["content"]
    return content if isinstance(content, str) else json.dumps(content)


# --------------------------------------------------------------------------
# The Inspector notes and the written defence
# --------------------------------------------------------------------------
def test_inspector_notes_cover_each_tool():
    assert INSPECTOR_FILE.exists(), "missing artifacts/step5/inspector.md"
    text = INSPECTOR_FILE.read_text()
    for name in ("search_logs", "read_file", "create_handoff"):
        assert name in text, f"inspector.md has nothing on {name}"
    assert "permission_denied" in text, "include one denied call as the Inspector showed it"


def test_where_the_check_lives_is_written_down():
    assert NOTE_FILE.exists(), "missing artifacts/step5/where-the-check-lives.md"
    text = NOTE_FILE.read_text()
    assert len(text.split()) >= 150, "at least 150 words"
    lowered = text.lower()
    assert "server" in lowered and "client" in lowered


# --------------------------------------------------------------------------
# The real run, through MCP
# --------------------------------------------------------------------------
def test_run_ended_with_an_answer(run):
    assert run["stopped"] == "answer"
    assert run["answer"].strip()


def test_every_tool_call_got_exactly_one_result_in_the_next_message(run):
    messages = run["messages"]
    for i, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        call_ids = [b["id"] for b in blocks(m, "tool_use")]
        if not call_ids:
            assert i == len(messages) - 1
            continue
        result_ids = [b.get("tool_use_id") for b in blocks(messages[i + 1], "tool_result")]
        assert result_ids == call_ids, f"turn {i}: results {result_ids} don't match calls {call_ids}"


def test_a_write_was_attempted_and_denied_by_the_server(run):
    writes = [(u, r) for u, r in calls_and_results(run) if u["name"] == "write_file"]
    assert writes, "the model never tried write_file: rerun, or make the prompt ask for the change directly"
    assert all(r and r.get("is_error") and "permission_denied" in content_text(r) for _, r in writes)


def test_the_model_escalated_after_the_denial(run):
    pairs = calls_and_results(run)
    names = [u["name"] for u, _ in pairs]
    first_write = names.index("write_file")
    assert "create_handoff" in names[first_write + 1:]
    handoff = next(r for u, r in pairs[first_write + 1:] if u["name"] == "create_handoff")
    assert handoff and not handoff.get("is_error")


def test_audit_log_recorded_the_denial_and_the_handoff(log):
    assert all("ts" in line and "event" in line for line in log)
    assert any(l["event"] == "permission_denied" and l["tool"] == "write_file" and l["level"] == "investigator"
               for l in log)
    handoffs = [l for l in log if l["event"] == "handoff"]
    assert handoffs and all(h["summary"].strip() and h["evidence"] for h in handoffs)


def test_log_and_run_describe_the_same_run(run, log):
    pairs = calls_and_results(run)
    handoff_text = " ".join(content_text(r) for u, r in pairs if u["name"] == "create_handoff" and r)
    logged_ids = [l["id"] for l in log if l["event"] == "handoff"]
    assert logged_ids and all(i in handoff_text for i in logged_ids), "agent.log should come from the same run"
    denied_in_run = sum(1 for _, r in pairs if r and r.get("is_error") and "permission_denied" in content_text(r))
    assert denied_in_run == sum(l["event"] == "permission_denied" for l in log), (
        "one audit line per denial: if the count is double, both client and server are logging"
    )


def test_artifacts_contain_no_api_keys():
    for path in (RUN_FILE, LOG_FILE, INSPECTOR_FILE, NOTE_FILE):
        if path.exists():
            assert not SECRET_PATTERN.search(path.read_text()), f"{path.name} looks like it contains an API key"

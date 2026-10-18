"""Step 4 acceptance tests, part 2: evidence from one real run of your gated loop. No API calls.

Run your loop once against Claude as an INVESTIGATOR, on a task that needs a write, with
AGENT_LOG pointing at a fresh file. The gate should deny the write, and the model should
escalate with create_handoff. Then commit both outputs:

    artifacts/step4/run.json     the dict run() returned (same shape as Step 3)
    artifacts/step4/agent.log    the audit log from that run, unedited

Suggested setup (any task works if it passes the checks below):
    rm -f artifacts/step4/agent.log
    AGENT_LOG=artifacts/step4/agent.log  (set before the run)
    prompt: "The PM wants each product's stock shown in search results.
             Find where search is implemented in shop/ and make the change."

The model doesn't know it's an investigator unless something tells it, so it will
usually try write_file first. If it goes straight to a handoff, run it again or make the
prompt more direct: the point is to capture a real denial followed by a real escalation.
Don't edit this file; if a test looks wrong, raise it.
"""
import json
import re
from pathlib import Path

import pytest

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts" / "step4"
RUN_FILE = ARTIFACTS / "run.json"
LOG_FILE = ARTIFACTS / "agent.log"
SECRET_PATTERN = re.compile(r"sk-[A-Za-z0-9_\-]{20,}|sk-ant-[A-Za-z0-9_\-]{10,}")


@pytest.fixture(scope="module")
def run():
    assert RUN_FILE.exists(), "missing artifacts/step4/run.json: save run()'s return value from a real run"
    return json.loads(RUN_FILE.read_text())


@pytest.fixture(scope="module")
def log():
    assert LOG_FILE.exists(), "missing artifacts/step4/agent.log: copy the AGENT_LOG from the same run"
    return [json.loads(line) for line in LOG_FILE.read_text().splitlines() if line.strip()]


def blocks(message, kind):
    content = message["content"]
    if isinstance(content, str):
        return []
    return [b for b in content if b.get("type") == kind]


def calls_and_results(run):
    """[(tool_use, tool_result)] pairs in order, matched by id."""
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


def test_run_ended_with_an_answer(run):
    assert run["stopped"] == "answer", "the saved run should end with a real answer, not the turn limit"
    assert run["answer"].strip()


def test_every_tool_call_got_exactly_one_result_in_the_next_message(run):
    messages = run["messages"]
    for i, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        call_ids = [b["id"] for b in blocks(m, "tool_use")]
        if not call_ids:
            assert i == len(messages) - 1, "an assistant turn without a tool call should end the run"
            continue
        result_ids = [b.get("tool_use_id") for b in blocks(messages[i + 1], "tool_result")]
        assert result_ids == call_ids, f"turn {i}: results {result_ids} don't match calls {call_ids}"


def test_a_write_was_attempted_and_denied(run):
    writes = [(u, r) for u, r in calls_and_results(run) if u["name"] == "write_file"]
    assert writes, "the model never tried write_file: rerun, or make the prompt ask for the change directly"
    assert all(r and r.get("is_error") and "permission_denied" in content_text(r) for _, r in writes), (
        "every write_file in an investigator run should come back as a permission_denied error"
    )


def test_the_model_escalated_after_the_denial(run):
    pairs = calls_and_results(run)
    names = [u["name"] for u, _ in pairs]
    first_write = names.index("write_file")
    assert "create_handoff" in names[first_write + 1:], "after the denial, the model should call create_handoff"
    handoff_result = next(r for u, r in pairs[first_write + 1:] if u["name"] == "create_handoff")
    assert handoff_result and not handoff_result.get("is_error"), "the handoff itself should succeed"


def test_audit_log_recorded_the_denial(log):
    assert all("ts" in line and "event" in line for line in log)
    denials = [line for line in log if line["event"] == "permission_denied"]
    assert any(d["tool"] == "write_file" and d["level"] == "investigator" for d in denials)


def test_audit_log_recorded_the_handoff(log):
    handoffs = [line for line in log if line["event"] == "handoff"]
    assert handoffs, "no handoff line in the audit log"
    for h in handoffs:
        assert h["summary"].strip() and h["requested_action"].strip()
        assert isinstance(h["evidence"], list) and h["evidence"]


def test_log_and_run_describe_the_same_run(run, log):
    returned_ids = set()
    for u, r in calls_and_results(run):
        if u["name"] == "create_handoff" and r and not r.get("is_error"):
            returned_ids.add(json.loads(r["content"])["id"])
    logged_ids = {line["id"] for line in log if line["event"] == "handoff"}
    assert returned_ids and returned_ids == logged_ids, "agent.log should come from the same run as run.json"
    denied_in_run = sum(
        1 for _, r in calls_and_results(run) if r and r.get("is_error") and "permission_denied" in content_text(r)
    )
    assert denied_in_run == sum(line["event"] == "permission_denied" for line in log)


def test_artifacts_contain_no_api_keys():
    for path in (RUN_FILE, LOG_FILE):
        if path.exists():
            assert not SECRET_PATTERN.search(path.read_text()), f"{path.name} looks like it contains an API key"

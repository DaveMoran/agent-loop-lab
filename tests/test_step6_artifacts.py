"""Step 6 acceptance tests, part 2: evidence from one real run, traced. No API calls.

Commit three things in artifacts/step6/:

    run.json        run_with_mcp()'s return value from one real Claude run against an INVESTIGATOR
                    server: the Step 4 and 5 scenario again, a denied write_file followed by
                    create_handoff.
    trace.jsonl    The trace file from that same run (AGENT_TRACE), unedited.
    view.txt        What `python -m agent.view_trace artifacts/step6/trace.jsonl` prints for it.

Suggested setup for the run:
    rm -f artifacts/step6/trace.jsonl
    AGENT_TRACE=artifacts/step6/trace.jsonl  (set before the run)
    the same prompt you used for Steps 4 and 5.

What a test can't judge (whether you'd regret what you left out of the record) is for the Step 6
quiz, which asks you to reconstruct a run from a pasted excerpt and spot the denied call.

Don't edit this file; if a test looks wrong, raise it.
"""
import json
import re
from datetime import datetime
from pathlib import Path

import pytest

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts" / "step6"
RUN_FILE = ARTIFACTS / "run.json"
TRACE_FILE = ARTIFACTS / "trace.jsonl"
VIEW_FILE = ARTIFACTS / "view.txt"
SECRET_PATTERN = re.compile(r"sk-[A-Za-z0-9_\-]{20,}|sk-ant-[A-Za-z0-9_\-]{10,}")


@pytest.fixture(scope="module")
def run():
    assert RUN_FILE.exists(), "missing artifacts/step6/run.json: save run_with_mcp()'s return value from a real run"
    return json.loads(RUN_FILE.read_text())


@pytest.fixture(scope="module")
def records():
    assert TRACE_FILE.exists(), "missing artifacts/step6/trace.jsonl: copy the AGENT_TRACE file from the same run"
    return [json.loads(line) for line in TRACE_FILE.read_text().splitlines() if line.strip()]


@pytest.fixture(scope="module")
def view_text():
    assert VIEW_FILE.exists(), "missing artifacts/step6/view.txt: save the viewer's output for this trace"
    return VIEW_FILE.read_text()


def all_calls(records):
    return [(r["turn"], c) for r in records for c in r["tool_calls"]]


# --------------------------------------------------------------------------
# The trace matches the run
# --------------------------------------------------------------------------
def test_run_ended_with_an_answer(run):
    assert run["stopped"] == "answer"
    assert run["answer"].strip()


def test_one_record_per_turn_of_a_single_run(run, records):
    assert [r["turn"] for r in records] == list(range(1, run["turns"] + 1))
    assert len({r["run_id"] for r in records}) == 1, "trace.jsonl should hold exactly this one run"


def test_the_trace_replays_the_run(run, records):
    replayed = [m for r in records for m in r["messages"]]
    assert replayed == run["messages"], "trace.jsonl and run.json should come from the same run"


def test_the_run_used_the_investigator_level(records):
    assert {r["level"] for r in records} == {"investigator"}


def test_every_call_result_is_what_the_model_was_shown(run, records):
    shown = {b["tool_use_id"]: b["content"]
             for m in run["messages"] if m["role"] == "user" and isinstance(m["content"], list)
             for b in m["content"] if b.get("type") == "tool_result"}
    calls = all_calls(records)
    assert calls and {c["id"] for _, c in calls} == set(shown)
    assert all(c["result"] == shown[c["id"]] for _, c in calls)


# --------------------------------------------------------------------------
# What a real run adds: real tokens, real latency, a real denial
# --------------------------------------------------------------------------
def test_every_turn_has_real_token_counts(records):
    for r in records:
        usage = r["usage"]
        assert isinstance(usage, dict), f"turn {r['turn']}: no usage. Where did it get dropped?"
        assert usage["input_tokens"] > 0 and usage["output_tokens"] > 0


def test_latencies_are_real_milliseconds(records):
    for r in records:
        assert 10 <= r["model_latency_ms"] < 600_000, (
            f"turn {r['turn']}: {r['model_latency_ms']} isn't a plausible API call in milliseconds")


def test_the_denied_write_and_the_handoff_are_in_the_trace(records):
    calls = all_calls(records)
    denied = [(t, c) for t, c in calls if c["name"] == "write_file" and c["status"] == "denied"]
    assert denied, "the trace should show write_file with status denied"
    first_denied_turn = denied[0][0]
    handoffs = [(t, c) for t, c in calls if c["name"] == "create_handoff" and c["status"] == "ok"]
    assert any(t > first_denied_turn for t, _ in handoffs), "the model should escalate after the denial"
    assert "permission_denied" in denied[0][1]["result"]


def test_only_the_denied_call_was_denied(records):
    denied = [c["name"] for _, c in all_calls(records) if c["status"] == "denied"]
    assert denied and set(denied) == {"write_file"}


def test_a_denied_call_cost_almost_nothing(records):
    denied = [c for _, c in all_calls(records) if c["status"] == "denied"]
    model_times = [r["model_latency_ms"] for r in records]
    assert all(c["latency_ms"] < min(model_times) for c in denied), "a refused call shouldn't take as long as a model call"


# --------------------------------------------------------------------------
# The viewer's output
# --------------------------------------------------------------------------
def test_view_shows_the_tools_and_the_denial(view_text):
    assert "write_file" in view_text and "create_handoff" in view_text
    assert "denied" in view_text.lower()


def test_view_matches_this_trace(records, view_text):
    first = records[0]["usage"]["input_tokens"]
    assert str(first) in view_text, "view.txt should be the viewer's output for this trace.jsonl"
    assert run_prompt_snippet(records) in view_text


def run_prompt_snippet(records):
    content = records[0]["messages"][0]["content"]
    text = content if isinstance(content, str) else json.dumps(content)
    return text.split()[0]


def test_artifacts_contain_no_api_keys():
    for path in (RUN_FILE, TRACE_FILE, VIEW_FILE):
        if path.exists():
            assert not SECRET_PATTERN.search(path.read_text()), f"{path.name} looks like it contains an API key"

"""Step 3 acceptance tests, part 2: evidence from one real run of your loop. No API calls.

Run your loop once against Claude, with the default tools, on a task that makes the model
choose between tools. Then save the dict that run() returned as JSON and commit it:

    artifacts/step3/run.json      {"answer": ..., "messages": [...], "turns": ..., "stopped": ...}

Suggested setup (any task works if it passes the checks below):
    SHOP_FAULTS=checkout_timeout python -m shop.simulate --n 200
    prompt: "Support says some checkouts are failing. Find out why, and which orders are affected."

Messages must be plain JSON (with an SDK: model_dump() each response before the loop stores
it, or make call_model return dicts). Don't edit this file; if a test looks wrong, raise it.
"""
import json
import re
from pathlib import Path

import pytest

RUN_FILE = Path(__file__).resolve().parents[1] / "artifacts" / "step3" / "run.json"
SECRET_PATTERN = re.compile(r"sk-[A-Za-z0-9_\-]{20,}|sk-ant-[A-Za-z0-9_\-]{10,}")


@pytest.fixture(scope="module")
def run():
    assert RUN_FILE.exists(), "missing artifacts/step3/run.json: save run()'s return value from a real run"
    return json.loads(RUN_FILE.read_text())


def blocks(message, kind):
    content = message["content"]
    if isinstance(content, str):
        return []
    return [b for b in content if b.get("type") == kind]


def test_run_ended_with_an_answer(run):
    assert set(run) >= {"answer", "messages", "turns", "stopped"}
    assert run["stopped"] == "answer", "the saved run should end with a real answer, not the turn limit"
    assert run["answer"].strip()


def test_history_starts_with_the_prompt_and_alternates(run):
    messages = run["messages"]
    assert messages[0]["role"] == "user" and isinstance(messages[0]["content"], str)
    roles = [m["role"] for m in messages]
    assert all(a != b for a, b in zip(roles, roles[1:])), f"roles must alternate: {roles}"
    assert roles[-1] == "assistant"


def test_turns_matches_the_model_calls_in_the_history(run):
    assert run["turns"] == sum(m["role"] == "assistant" for m in run["messages"])


def test_every_tool_call_got_exactly_one_result_in_the_next_message(run):
    messages = run["messages"]
    for i, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        call_ids = [b["id"] for b in blocks(m, "tool_use")]
        if not call_ids:
            assert i == len(messages) - 1, "an assistant turn without a tool call should end the run"
            continue
        result_ids = [b["tool_use_id"] for b in blocks(messages[i + 1], "tool_result")]
        assert result_ids == call_ids, f"turn {i}: results {result_ids} don't match calls {call_ids}"


def test_the_model_had_to_choose_between_tools(run):
    used = {b["name"] for m in run["messages"] if m["role"] == "assistant" for b in blocks(m, "tool_use")}
    assert len(used) >= 2, f"only used {used or 'no tools'}: pick a task that needs more than one tool"
    assert used <= {"search_logs", "read_file", "run_query"}


def test_final_turn_is_text_and_matches_the_answer(run):
    last = run["messages"][-1]
    assert not blocks(last, "tool_use")
    text = "".join(b.get("text", "") for b in blocks(last, "text"))
    assert text.strip() and text.strip() == run["answer"].strip()


def test_run_contains_no_api_keys():
    assert not SECRET_PATTERN.search(RUN_FILE.read_text()), "run.json looks like it contains an API key"

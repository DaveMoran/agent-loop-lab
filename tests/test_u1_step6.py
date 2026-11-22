"""Unit 1, Step 6 acceptance tests: create_handoff tells the model what was wrong. No API calls.

Don't edit this file; if a test looks wrong, raise it and we'll change it together.

This closes the carry-forward from the Step 5 review. Through MCP, a create_handoff call with
an empty evidence list, a blank summary or a blank requested_action reaches the model only as
the SDK's generic "Error executing tool create_handoff". The model can't fix a call when it
isn't told what was wrong with it.

What these tests pin (the contract), and nothing more:

    create_handoff with an empty evidence list, a blank summary (empty or whitespace only) or a
    blank requested_action comes back to the model with is_error true and a message that names
    the problem ("evidence", "summary" or "requested"). The SDK may keep its generic prefix;
    the reason has to follow it.

    The message does not echo the call's arguments back to the model (Step 4's rule: arguments
    belong in the audit log, not in the model-facing error).

    No handoff is written to the audit log for a rejected call.

    The same holds through loop.run, so the two loops tell the model the same thing.

Left open on purpose: where the conversion happens (in the tool, in the server, or in a helper
both use), the exact wording, and whether other tools get the same treatment.
"""
import copy
import json
from pathlib import Path

import pytest

PROMPT = "Checkouts are failing. Hand this off to someone who can write."
MARKER = "SUMMARY-MARKER-7f3a"

CASES = [
    pytest.param({"summary": MARKER, "evidence": [], "requested_action": "restart payments"},
                 "evidence", id="empty-evidence"),
    pytest.param({"summary": "   ", "evidence": [MARKER], "requested_action": "restart payments"},
                 "summary", id="blank-summary"),
    pytest.param({"summary": MARKER, "evidence": ["payment_timeout x53"], "requested_action": ""},
                 "requested", id="blank-requested-action"),
]


@pytest.fixture(autouse=True)
def audit_log(tmp_path, monkeypatch):
    path = tmp_path / "agent.log"
    monkeypatch.setenv("AGENT_LOG", str(path))
    return path


class ScriptedModel:
    """Canned responses in order. Works as call_model(messages) and as call_model(messages, tools)."""

    def __init__(self, *responses):
        self.responses = list(responses)

    def __call__(self, messages, tools=None):
        if not self.responses:
            raise AssertionError("the loop called the model more times than scripted")
        return copy.deepcopy(self.responses.pop(0))


def asks_for_handoff(arguments):
    return {"id": "msg_x", "type": "message", "role": "assistant", "model": "fake",
            "stop_reason": "tool_use",
            "content": [{"type": "tool_use", "id": "tu_1", "name": "create_handoff", "input": arguments}],
            "usage": {"input_tokens": 10, "output_tokens": 5}}


def answers(text):
    return {"id": "msg_y", "type": "message", "role": "assistant", "model": "fake",
            "stop_reason": "end_turn", "content": [{"type": "text", "text": text}],
            "usage": {"input_tokens": 10, "output_tokens": 5}}


@pytest.fixture(params=["loop", "mcp"])
def drive(request):
    """Run a scripted model through either loop, as an investigator."""

    def go(model):
        if request.param == "loop":
            from agent import loop
            from agent.permissions import Context

            return loop.run(PROMPT, model, ctx=Context("investigator"))
        from agent import mcp_client, mcp_server

        return mcp_client.run_with_mcp(PROMPT, model, mcp_server.build_server("investigator"))

    return go


def the_result_block(messages):
    blocks = [b for m in messages if m["role"] == "user" and isinstance(m["content"], list)
              for b in m["content"] if b.get("type") == "tool_result"]
    assert len(blocks) == 1, f"expected one tool_result, found {len(blocks)}"
    return blocks[0]


def as_text(content):
    """A tool_result's content may be a string or a list of text blocks."""
    if isinstance(content, str):
        return content
    return " ".join(b.get("text", "") for b in content if isinstance(b, dict))


def handoffs_written(audit_log):
    if not Path(audit_log).exists():
        return []
    records = [json.loads(line) for line in Path(audit_log).read_text().splitlines() if line.strip()]
    return [r for r in records if r.get("event") == "handoff"]


@pytest.mark.parametrize("arguments, word", CASES)
def test_the_model_is_told_what_was_wrong(drive, arguments, word):
    result = drive(ScriptedModel(asks_for_handoff(arguments), answers("ok")))
    block = the_result_block(result["messages"])
    text = as_text(block["content"])
    assert block.get("is_error") is True, "a rejected handoff must come back as an error"
    assert text.strip() != "Error executing tool create_handoff", \
        "the model is only told the tool failed, not why"
    assert word in text.lower(), f"the error should name the problem ({word!r}), got {text!r}"


@pytest.mark.parametrize("arguments, word", CASES)
def test_the_error_does_not_echo_the_arguments(drive, arguments, word):
    result = drive(ScriptedModel(asks_for_handoff(arguments), answers("ok")))
    text = as_text(the_result_block(result["messages"])["content"])
    assert MARKER not in text, "the model-facing error echoes the call's arguments"


@pytest.mark.parametrize("arguments, word", CASES)
def test_a_rejected_handoff_is_not_written(drive, audit_log, arguments, word):
    drive(ScriptedModel(asks_for_handoff(arguments), answers("ok")))
    assert handoffs_written(audit_log) == [], "a rejected handoff still reached the audit log"


def test_a_valid_handoff_still_works(drive, audit_log):
    arguments = {"summary": "Checkout fails on orders over $100",
                 "evidence": ["53 payment_timeout errors, all over $100"],
                 "requested_action": "Raise the payment timeout in shop/app.py"}
    result = drive(ScriptedModel(asks_for_handoff(arguments), answers("Handed off.")))
    block = the_result_block(result["messages"])
    assert not block.get("is_error"), f"a valid handoff was rejected: {as_text(block['content'])!r}"
    assert len(handoffs_written(audit_log)) == 1

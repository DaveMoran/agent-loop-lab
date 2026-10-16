"""Step 2 acceptance tests, part 2: the evidence from your real round trips. No API calls.

These check the files your own code writes during one real, complete round trip per
provider, which you commit. They replace live API tests: proof that a real round trip
happened, without flaky or paid calls in the suite.

A complete round trip is: request -> model asks for the tool -> YOUR code runs it ->
second request carrying the result -> model gives a final answer. So per provider:

    artifacts/step2/openai_tool_request.json       1st response: OpenAI (Responses API) asks for the tool
    artifacts/step2/openai_second_request.json     the request body you sent on the 2nd call
    artifacts/step2/openai_final_response.json     2nd response: OpenAI's final answer
    artifacts/step2/anthropic_tool_request.json    1st response: Claude (Messages API) asks for the tool
    artifacts/step2/anthropic_second_request.json  the request body you sent on the 2nd call
    artifacts/step2/anthropic_final_response.json  2nd response: Claude's final answer
    artifacts/step2/comparison.md                  your short side-by-side comparison

Responses ("*_tool_request", "*_final_response") are the whole raw response object dumped
to JSON (with an SDK: model_dump()), not trimmed copies. "*_second_request" is the body
you sent as a JSON object: for OpenAI the keyword arguments you passed (at least "input",
plus "previous_response_id" if you chain that way); for Claude at least "messages".
Leaving out "tools" and the model name is fine. All six files must come from the same run.

Don't edit this file; if a test looks wrong, raise it.
"""
import importlib
import json
import re
from pathlib import Path

import pytest

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts" / "step2"
OPENAI_FILE = ARTIFACTS / "openai_tool_request.json"
ANTHROPIC_FILE = ARTIFACTS / "anthropic_tool_request.json"
OPENAI_SECOND = ARTIFACTS / "openai_second_request.json"
ANTHROPIC_SECOND = ARTIFACTS / "anthropic_second_request.json"
OPENAI_FINAL = ARTIFACTS / "openai_final_response.json"
ANTHROPIC_FINAL = ARTIFACTS / "anthropic_final_response.json"
COMPARISON_FILE = ARTIFACTS / "comparison.md"
ALL_JSON = (OPENAI_FILE, ANTHROPIC_FILE, OPENAI_SECOND, ANTHROPIC_SECOND, OPENAI_FINAL, ANTHROPIC_FINAL)

TOOL_ARGS = {"query", "level", "limit"}
SECRET_PATTERN = re.compile(r"sk-[A-Za-z0-9_\-]{20,}|sk-ant-[A-Za-z0-9_\-]{10,}")


def load_json(path):
    assert path.exists(), f"missing artifacts/step2/{path.name}: save it from a real run"
    return json.loads(path.read_text())


def openai_function_call(data):
    assert "output" in data, "expected a Responses API response (has 'output'), not Chat Completions"
    items = [i for i in data["output"] if i.get("type") == "function_call"]
    assert items, "the OpenAI response has no function_call output item"
    return items[0]


def claude_tool_use(data):
    blocks = [b for b in data.get("content", []) if b.get("type") == "tool_use"]
    assert blocks, "the Claude response has no tool_use content block"
    return blocks[0]


@pytest.fixture
def small_log(tmp_path, monkeypatch):
    path = tmp_path / "app.log"
    path.write_text(
        json.dumps({"ts": 1.0, "level": "ERROR", "event": "payment_timeout", "total": 162.9}) + "\n"
    )
    monkeypatch.setenv("SHOP_LOG", str(path))
    return path


def test_openai_payload_is_a_raw_tool_request():
    data = load_json(OPENAI_FILE)
    assert "id" in data and "model" in data, "looks trimmed: save the whole response object"
    call = openai_function_call(data)
    assert call["name"] == "search_logs"
    assert call["call_id"]
    assert isinstance(call["arguments"], str), "OpenAI sends arguments as a JSON string; keep it as returned"
    assert isinstance(json.loads(call["arguments"]), dict)


def test_anthropic_payload_is_a_raw_tool_request():
    data = load_json(ANTHROPIC_FILE)
    assert "id" in data and "model" in data, "looks trimmed: save the whole response object"
    assert data.get("stop_reason") == "tool_use"
    block = claude_tool_use(data)
    assert block["name"] == "search_logs"
    assert block["id"]
    assert isinstance(block["input"], dict), "Claude sends input as an object; keep it as returned"


def test_your_extractors_read_your_real_payloads():
    oai = importlib.import_module("agent.openai_raw")
    claude = importlib.import_module("agent.claude_raw")
    [o] = oai.extract_tool_calls(load_json(OPENAI_FILE))[:1]
    [c] = claude.extract_tool_calls(load_json(ANTHROPIC_FILE))[:1]
    assert o["name"] == c["name"] == "search_logs"


def test_the_arguments_each_model_chose_actually_run_against_the_tool(small_log):
    search_logs = importlib.import_module("agent.tools").search_logs
    openai_args = json.loads(openai_function_call(load_json(OPENAI_FILE))["arguments"])
    claude_args = claude_tool_use(load_json(ANTHROPIC_FILE))["input"]
    for args in (openai_args, claude_args):
        assert set(args) <= TOOL_ARGS, f"model sent arguments your tool doesn't take: {set(args) - TOOL_ARGS}"
        assert isinstance(search_logs(**args), list)


def test_saved_payloads_contain_no_api_keys():
    for path in ALL_JSON:
        assert path.exists(), f"missing artifacts/step2/{path.name}"
        assert not SECRET_PATTERN.search(path.read_text()), f"{path.name} looks like it contains an API key"


def test_comparison_covers_both_providers_and_the_real_differences():
    assert COMPARISON_FILE.exists(), "missing artifacts/step2/comparison.md"
    text = COMPARISON_FILE.read_text().lower()
    assert len(text.split()) >= 40, "write at least a short paragraph or table"
    assert "openai" in text
    assert "claude" in text or "anthropic" in text
    # the identifiers each side uses for a tool request, its arguments, and the link back
    assert "tool_use" in text and "function_call" in text
    assert "arguments" in text and "input" in text
    assert "call_id" in text and "tool_use_id" in text


# ==========================================================================
# The second half of the round trip: what you sent back, and what came back
# ==========================================================================
def tool_output_rows(output_str, args):
    """The tool output you sent back: a JSON list of log records that honours the level asked for."""
    assert isinstance(output_str, str), "send the tool output back as a JSON string"
    rows = json.loads(output_str)
    assert isinstance(rows, list) and all(isinstance(r, dict) for r in rows), "expected search_logs' list of records"
    assert rows, (
        "your tool returned nothing, so the model answered from no data. Check the arguments the "
        "model chose against your log (and your schema). If empty really is right, use a prompt that finds something."
    )
    if args.get("level"):
        assert all(r.get("level", "").lower() == args["level"].lower() for r in rows), "records don't match the level"
    return rows


def test_openai_second_request_sends_the_result_back_linked_to_the_call():
    first = load_json(OPENAI_FILE)
    call = openai_function_call(first)
    body = load_json(OPENAI_SECOND)
    assert isinstance(body.get("input"), list), "save the body you sent: it needs the 'input' list"
    outputs = [i for i in body["input"] if isinstance(i, dict) and i.get("type") == "function_call_output"]
    assert outputs, "the second request has no function_call_output item"
    assert outputs[0]["call_id"] == call["call_id"], "the result must reference the call's call_id (not the fc_ id)"
    # The model has to see its own call again: either replayed in input, or chained by response id.
    replayed = any(
        isinstance(i, dict) and i.get("type") == "function_call" and i.get("call_id") == call["call_id"]
        for i in body["input"]
    )
    assert replayed or body.get("previous_response_id") == first["id"], (
        "the model can't match a result to a call it can't see: replay the function_call item in 'input', "
        "or chain with previous_response_id"
    )
    tool_output_rows(outputs[0]["output"], json.loads(call["arguments"]))


def test_anthropic_second_request_sends_the_result_back_linked_to_the_call():
    first = load_json(ANTHROPIC_FILE)
    tool_use = claude_tool_use(first)
    body = load_json(ANTHROPIC_SECOND)
    messages = body.get("messages")
    assert isinstance(messages, list) and len(messages) >= 3, "expected the prompt, the assistant turn and the result"
    assistant, result = messages[-2], messages[-1]
    assert assistant["role"] == "assistant", "the model's tool-use turn must come right before the result"
    assert isinstance(assistant["content"], list)
    assert [b["type"] for b in assistant["content"]] == [b["type"] for b in first["content"]], (
        "replay the assistant turn exactly as returned, every block (including thinking) in the same order"
    )
    assert result["role"] == "user", "tool results go back in a user message"
    blocks = [b for b in result["content"] if b.get("type") == "tool_result"]
    assert blocks, "the last message has no tool_result block"
    assert blocks[0]["tool_use_id"] == tool_use["id"], "tool_use_id must match the tool_use block's id"
    tool_output_rows(blocks[0]["content"], tool_use["input"])


def test_openai_final_response_is_an_answer():
    data = load_json(OPENAI_FINAL)
    assert "id" in data and "model" in data, "looks trimmed: save the whole response object"
    assert data["id"] != load_json(OPENAI_FILE)["id"], "this is the first response again; save the second one"
    assert not [i for i in data.get("output", []) if i.get("type") == "function_call"], (
        "the model asked for another tool call; for Step 2, prompt so one call is enough"
    )
    texts = [
        c.get("text", "")
        for i in data.get("output", []) if i.get("type") == "message"
        for c in i.get("content", []) if c.get("type") == "output_text"
    ]
    assert "".join(texts).strip(), "no final text answer in the OpenAI response"


def test_anthropic_final_response_is_an_answer():
    data = load_json(ANTHROPIC_FINAL)
    assert "id" in data and "model" in data, "looks trimmed: save the whole response object"
    assert data["id"] != load_json(ANTHROPIC_FILE)["id"], "this is the first response again; save the second one"
    assert data.get("stop_reason") == "end_turn", "the model should finish its turn, not ask for another tool"
    text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
    assert text.strip(), "no final text answer in the Claude response"

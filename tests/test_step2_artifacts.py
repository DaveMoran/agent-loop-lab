"""Step 2 acceptance tests, part 2: the evidence from your real round trips. No API calls.

These check the files your own code writes after a real run against each provider,
which you commit. They replace live API tests: proof that a real round trip happened,
without flaky or paid calls in the suite.

    artifacts/step2/openai_tool_request.json     the raw OpenAI (Responses API) response that asks for the tool
    artifacts/step2/anthropic_tool_request.json  the raw Claude (Messages API) response that asks for the tool
    artifacts/step2/comparison.md                your short side-by-side comparison of the two

"Raw" means the whole response object dumped to JSON (with an SDK: model_dump()), not a
trimmed copy. Don't edit this file; if a test looks wrong, raise it.
"""
import importlib
import json
import re
from pathlib import Path

import pytest

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts" / "step2"
OPENAI_FILE = ARTIFACTS / "openai_tool_request.json"
ANTHROPIC_FILE = ARTIFACTS / "anthropic_tool_request.json"
COMPARISON_FILE = ARTIFACTS / "comparison.md"

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
    for path in (OPENAI_FILE, ANTHROPIC_FILE):
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

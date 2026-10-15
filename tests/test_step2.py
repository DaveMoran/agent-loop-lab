"""Step 2 acceptance tests, part 1: the code contract. No API calls.

Part 2 (the evidence from your real runs) is tests/test_step2_artifacts.py.
Don't edit these files; if a test looks wrong, raise it and we'll change it together.

What these tests pin (the contract), and nothing more:

agent/tools.py
    search_logs(query: str | None, level: str | None = None, limit: int = 20) -> list[dict]
        - Reads the log at shop.logs.log_path() at call time (honours SHOP_LOG).
        - query: case-insensitive substring match on the raw log line, so it hits
          field values too, not just the event name. None or "" matches all.
        - level: case-insensitive exact match on the record's "level". None = any.
        - limit: return at most `limit` matches, the MOST RECENT ones, in file order.
          limit=0 returns [].
        - Returns the parsed records and prints nothing.
        - Skips lines that aren't valid JSON. Raises FileNotFoundError if the log is missing.
    SEARCH_LOGS_PARAMETERS: dict
        - The JSON Schema for search_logs' arguments, shared by both providers.

agent/openai_raw.py   (OpenAI Responses API)
agent/claude_raw.py   (Anthropic Messages API)
    tool_definition() -> dict
        - search_logs described in that provider's tool format.
    extract_tool_calls(response: dict) -> list[dict]
        - Takes the raw response JSON as a dict (with an SDK: response.model_dump()).
        - Returns [{"id": ..., "name": ..., "arguments": {...}}, ...], where "id" is the
          value the tool result must reference, and "arguments" is already a dict.
        - Returns [] when the model answered without asking for a tool.
    tool_result(call_id: str, result) -> dict
        - The thing you send back so the model can see the result. The result is
          JSON-serialised into a string.

How you make the request, run the round trip, print payloads, structure the
modules beyond these names, or choose models is up to you.

The provider responses below are hand-built from each provider's documented
format, not recorded. When you save your real payloads (part 2), compare them with
these. If a field differs, tell me, because that is quiz material.
"""
import importlib
import json

import pytest


# --------------------------------------------------------------------------
# Imports happen inside fixtures so each test fails on its own while you build,
# instead of the whole file erroring at collection.
# --------------------------------------------------------------------------
@pytest.fixture
def tools():
    return importlib.import_module("agent.tools")


@pytest.fixture
def oai():
    return importlib.import_module("agent.openai_raw")


@pytest.fixture
def claude():
    return importlib.import_module("agent.claude_raw")


# --------------------------------------------------------------------------
# A small, known log, written in the same JSON-lines format as shop/logs.py
# --------------------------------------------------------------------------
LOG_RECORDS = [
    {"ts": 1.0, "level": "INFO", "event": "search", "query": "Socks", "duration_ms": 12},
    {"ts": 2.0, "level": "ERROR", "event": "payment_timeout", "user_id": 13, "total": 162.9},
    {"ts": 3.0, "level": "WARN", "event": "out_of_stock", "product_id": 4},
    {"ts": 4.0, "level": "INFO", "event": "checkout_ok", "user_id": 2, "total": 40.0},
    {"ts": 5.0, "level": "ERROR", "event": "payment_timeout", "user_id": 1, "total": 227.86},
    {"ts": 6.0, "level": "WARN", "event": "search", "query": "lamp", "duration_ms": 2400},
    {"ts": 7.0, "level": "ERROR", "event": "payment_timeout", "user_id": 9, "total": 131.5},
]


@pytest.fixture
def log_file(tmp_path, monkeypatch):
    path = tmp_path / "app.log"
    path.write_text("".join(json.dumps(r) + "\n" for r in LOG_RECORDS))
    monkeypatch.setenv("SHOP_LOG", str(path))
    return path


# ==========================================================================
# 1. The tool itself: search_logs
# ==========================================================================
def test_search_logs_filters_by_query(tools, log_file):
    rows = tools.search_logs("payment_timeout")
    assert [r["user_id"] for r in rows] == [13, 1, 9]


def test_search_logs_returns_parsed_records_and_prints_nothing(tools, log_file, capsys):
    rows = tools.search_logs("payment_timeout")
    assert rows[0] == LOG_RECORDS[1]
    assert capsys.readouterr().out == "", "the tool returns data; printing is the caller's job"


def test_search_logs_query_is_case_insensitive(tools, log_file):
    assert len(tools.search_logs("PAYMENT_TIMEOUT")) == 3


def test_search_logs_query_matches_field_values(tools, log_file):
    rows = tools.search_logs("socks")
    assert [r["query"] for r in rows] == ["Socks"]


def test_search_logs_none_or_empty_query_matches_everything(tools, log_file):
    assert tools.search_logs(None, limit=100) == LOG_RECORDS
    assert tools.search_logs("", limit=100) == LOG_RECORDS


def test_search_logs_filters_by_level(tools, log_file):
    rows = tools.search_logs("", level="warn")
    assert [r["ts"] for r in rows] == [3.0, 6.0]


def test_search_logs_combines_query_and_level(tools, log_file):
    rows = tools.search_logs("search", level="WARN")
    assert len(rows) == 1 and rows[0]["duration_ms"] == 2400


def test_search_logs_limit_keeps_most_recent_in_file_order(tools, log_file):
    rows = tools.search_logs("payment_timeout", limit=2)
    assert [r["user_id"] for r in rows] == [1, 9]


def test_search_logs_limit_zero_returns_nothing(tools, log_file):
    assert tools.search_logs("payment_timeout", limit=0) == []


def test_search_logs_no_match_returns_empty_list(tools, log_file):
    assert tools.search_logs("refund_failed") == []


def test_search_logs_skips_malformed_lines(tools, log_file):
    with log_file.open("a") as f:
        f.write("this is not json payment_timeout\n")
    assert len(tools.search_logs("payment_timeout")) == 3


def test_search_logs_missing_file_raises(tools, tmp_path, monkeypatch):
    monkeypatch.setenv("SHOP_LOG", str(tmp_path / "nope.log"))
    with pytest.raises(FileNotFoundError):
        tools.search_logs("anything")


def test_parameters_schema_describes_the_arguments(tools):
    schema = tools.SEARCH_LOGS_PARAMETERS
    assert schema["type"] == "object"
    assert set(schema["properties"]) == {"query", "level", "limit"}
    assert schema.get("required") == ["query"]
    for name, prop in schema["properties"].items():
        assert prop.get("description"), f"{name} needs a description: the model reads it"


# ==========================================================================
# 2. OpenAI (Responses API)
# ==========================================================================
OPENAI_TOOL_RESPONSE = {
    "id": "resp_67ccd2bed1ec8190b14f964abc054267",
    "object": "response",
    "status": "completed",
    "output": [
        {
            "type": "reasoning",
            "id": "rs_67ccd2bf17f0819081ff3bb2cf6508e6",
            "summary": [],
        },
        {
            "type": "function_call",
            "id": "fc_67ccd2bf3a8c8190a9d9d4b8f9c5c1e2",
            "call_id": "call_9Xk2LqP0aZ7mQ1",
            "name": "search_logs",
            "arguments": "{\"query\":\"payment_timeout\",\"level\":\"ERROR\",\"limit\":5}",
            "status": "completed",
        },
    ],
}

OPENAI_TWO_CALLS_RESPONSE = {
    "id": "resp_2",
    "object": "response",
    "status": "completed",
    "output": [
        {
            "type": "function_call",
            "id": "fc_a",
            "call_id": "call_a",
            "name": "search_logs",
            "arguments": "{\"query\":\"payment_timeout\"}",
            "status": "completed",
        },
        {
            "type": "function_call",
            "id": "fc_b",
            "call_id": "call_b",
            "name": "search_logs",
            "arguments": "{\"query\":\"search\",\"level\":\"WARN\"}",
            "status": "completed",
        },
    ],
}

OPENAI_TEXT_RESPONSE = {
    "id": "resp_3",
    "object": "response",
    "status": "completed",
    "output": [
        {
            "type": "message",
            "id": "msg_1",
            "role": "assistant",
            "status": "completed",
            "content": [
                {"type": "output_text", "text": "Three payment timeouts, all over $100.", "annotations": []}
            ],
        }
    ],
}


def test_openai_tool_definition_shape(oai, tools):
    d = oai.tool_definition()
    assert d["type"] == "function"
    assert d["name"] == "search_logs"
    assert d.get("description")
    assert d["parameters"] == tools.SEARCH_LOGS_PARAMETERS
    assert "function" not in d, "that's the Chat Completions shape; Responses is flat"


def test_openai_extract_uses_call_id_and_parses_arguments(oai):
    calls = oai.extract_tool_calls(OPENAI_TOOL_RESPONSE)
    assert calls == [
        {
            "id": "call_9Xk2LqP0aZ7mQ1",  # call_id, NOT the fc_ item id
            "name": "search_logs",
            "arguments": {"query": "payment_timeout", "level": "ERROR", "limit": 5},
        }
    ]


def test_openai_extract_handles_parallel_calls(oai):
    calls = oai.extract_tool_calls(OPENAI_TWO_CALLS_RESPONSE)
    assert [c["id"] for c in calls] == ["call_a", "call_b"]
    assert calls[1]["arguments"] == {"query": "search", "level": "WARN"}


def test_openai_extract_returns_empty_when_model_just_answers(oai):
    assert oai.extract_tool_calls(OPENAI_TEXT_RESPONSE) == []


def test_openai_tool_result_shape(oai):
    result = [{"event": "payment_timeout", "total": 162.9}]
    item = oai.tool_result("call_9Xk2LqP0aZ7mQ1", result)
    assert item["type"] == "function_call_output"
    assert item["call_id"] == "call_9Xk2LqP0aZ7mQ1"
    assert isinstance(item["output"], str)
    assert json.loads(item["output"]) == result


# ==========================================================================
# 3. Claude (Messages API)
# ==========================================================================
CLAUDE_TOOL_RESPONSE = {
    "id": "msg_01Aq9w938a90dw8q",
    "type": "message",
    "role": "assistant",
    "model": "claude-example",
    "stop_reason": "tool_use",
    "content": [
        {"type": "text", "text": "I'll search the logs for payment errors."},
        {
            "type": "tool_use",
            "id": "toolu_01A09q90qw90lq917835lq9",
            "name": "search_logs",
            "input": {"query": "payment_timeout", "level": "ERROR", "limit": 5},
        },
    ],
}

CLAUDE_TWO_CALLS_RESPONSE = {
    "id": "msg_2",
    "type": "message",
    "role": "assistant",
    "model": "claude-example",
    "stop_reason": "tool_use",
    "content": [
        {"type": "tool_use", "id": "toolu_a", "name": "search_logs", "input": {"query": "payment_timeout"}},
        {"type": "tool_use", "id": "toolu_b", "name": "search_logs", "input": {"query": "search", "level": "WARN"}},
    ],
}

CLAUDE_TEXT_RESPONSE = {
    "id": "msg_3",
    "type": "message",
    "role": "assistant",
    "model": "claude-example",
    "stop_reason": "end_turn",
    "content": [{"type": "text", "text": "Three payment timeouts, all over $100."}],
}


def test_claude_tool_definition_shape(claude, tools):
    d = claude.tool_definition()
    assert d["name"] == "search_logs"
    assert d.get("description")
    assert d["input_schema"] == tools.SEARCH_LOGS_PARAMETERS
    assert "parameters" not in d and "type" not in d, "that's OpenAI's shape"


def test_claude_extract_skips_text_and_keeps_input(claude):
    calls = claude.extract_tool_calls(CLAUDE_TOOL_RESPONSE)
    assert calls == [
        {
            "id": "toolu_01A09q90qw90lq917835lq9",
            "name": "search_logs",
            "arguments": {"query": "payment_timeout", "level": "ERROR", "limit": 5},
        }
    ]


def test_claude_extract_handles_parallel_calls(claude):
    calls = claude.extract_tool_calls(CLAUDE_TWO_CALLS_RESPONSE)
    assert [c["id"] for c in calls] == ["toolu_a", "toolu_b"]


def test_claude_extract_returns_empty_when_model_just_answers(claude):
    assert claude.extract_tool_calls(CLAUDE_TEXT_RESPONSE) == []


def test_claude_tool_result_shape(claude):
    result = [{"event": "payment_timeout", "total": 162.9}]
    msg = claude.tool_result("toolu_01A09q90qw90lq917835lq9", result)
    assert msg["role"] == "user"
    [block] = msg["content"]
    assert block["type"] == "tool_result"
    assert block["tool_use_id"] == "toolu_01A09q90qw90lq917835lq9"
    assert isinstance(block["content"], str)
    assert json.loads(block["content"]) == result

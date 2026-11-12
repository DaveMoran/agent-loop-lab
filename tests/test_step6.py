"""Step 6 acceptance tests, part 1: trace every run, and read it back. No API calls.

Part 2 (evidence from one real run) is tests/test_step6_artifacts.py.
Don't edit these files; if a test looks wrong, raise it and we'll change it together.

These tests drive both of your loops, `loop.run` and `run_with_mcp`, with a scripted model, then
read the trace file they leave behind. They check what a trace has to hold so that you can rebuild
a run from it without calling the model again. Where the code that assembles a record lives is
up to you; the tests only look at the file.

What these tests pin (the contract), and nothing more:

agent/trace.py  (new)
    write_turn(record: dict) -> None
        - Appends one JSON line, {"ts": <ISO-8601 UTC string>, **record}, to the file named by the
          AGENT_TRACE environment variable, read at call time. Default: logs/trace.jsonl in the
          repo. Creates missing directories. The same shape of writer as write_to_audit_log, but a
          separate file: the audit log records decisions, the trace records turns.
    read_trace(path) -> list[dict]
        - Parses a JSON Lines file into records, in file order. Blank lines are skipped.

Both loops write exactly one record per model turn (the turn the model answers on included), with
the same shape, through the same writer:

    ts                  str, ISO 8601 (added by write_turn)
    run_id              str. The same for every turn of one run; a new run gets a new id.
    turn                int, 1-based, consecutive
    level               "investigator" or "operator": the level the run used. loop.run has a
                        Context. run_with_mcp doesn't, so ask what the client can learn from
                        the server it is talking to.
    stop_reason         str, the model's own stop_reason for that turn
    usage               dict or None. Copied from the response's "usage" (it must carry
                        input_tokens and output_tokens; extra keys are fine). None when the
                        response has no usage. A response may also be a pydantic-style object
                        with .model_dump(), as the loops already allow.
    model_latency_ms    number >= 0: how long the model call took, in milliseconds
    messages            list: the messages appended to the history during this turn. Turn 1
                        includes the user's prompt. Put the records' lists end to end and you
                        get exactly the "messages" the run returned: that is the replay test.
    tool_calls          list, in call order, empty when the model asked for no tools. Each:
                            id, name        str
                            arguments       dict, as the model sent them
                            status          "ok", "error" or "denied"
                            result          str, exactly the content of that call's tool_result
                                            block in messages (what the model was shown)
                            latency_ms      number >= 0, how long the tool took. A denied call
                                            never runs the tool.
    "denied" means the permission gate refused it, in either loop. Through MCP the client only
    sees an error result with text in it, so it has to recognise the server's denial payload.
    Any other failure is "error".

agent/view_trace.py  (new)
    render_run(records: list[dict]) -> str
        - Takes the records of ONE run and returns a readable conversation: the prompt, what the
          assistant said, each tool call with its arguments, status and result, the final answer,
          and per turn the token counts. A denied call must be easy to spot: the word "denied"
          appears for it, and doesn't appear for a run that had none.
    python -m agent.view_trace [PATH] [--run RUN_ID]
        - Prints one run from a trace file to stdout. PATH defaults to AGENT_TRACE (then to the
          default file). Without --run it prints the run of the last record in the file. An
          unknown run id or a missing file exits non-zero with a message, not a traceback.

agent/mcp_server.py  (carry-forward from the Step 5 review)
    - create_handoff with an empty evidence list, a blank summary or a blank requested_action
      comes back with is_error true and a message that names the problem, not just the generic
      "Error executing tool create_handoff". (The SDK puts that prefix on every tool failure; the
      reason has to follow it.) Nothing is written to the audit log.

Left open on purpose: how run_id is made, where records are built (so two loops don't duplicate
the code), what extra fields you add, how the viewer lays things out, and what a loop should do
if writing the trace fails.
"""
import copy
import importlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PROMPT = "Checkouts are failing. Look into it and fix shop/app.py."
USAGE = {"input_tokens": 1234, "output_tokens": 56}


# --------------------------------------------------------------------------
# Fixtures and helpers
# --------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def audit_log(tmp_path, monkeypatch):
    path = tmp_path / "agent.log"
    monkeypatch.setenv("AGENT_LOG", str(path))
    return path


@pytest.fixture(autouse=True)
def trace_file(tmp_path, monkeypatch):
    path = tmp_path / "trace.jsonl"
    monkeypatch.setenv("AGENT_TRACE", str(path))
    return path


@pytest.fixture
def shop_log(tmp_path, monkeypatch):
    path = tmp_path / "app.log"
    lines = [
        {"ts": 1, "level": "INFO", "event": "search", "query": "mug", "duration_ms": 12},
        {"ts": 2, "level": "ERROR", "event": "payment_timeout", "order_total": 142.5},
    ]
    path.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    monkeypatch.setenv("SHOP_LOG", str(path))
    return path


@pytest.fixture
def trace():
    return importlib.import_module("agent.trace")


@pytest.fixture
def view():
    return importlib.import_module("agent.view_trace")


class ScriptedModel:
    """Canned responses in order. Works as call_model(messages) and as call_model(messages, tools)."""

    def __init__(self, *responses, delay=0.0):
        self.responses = list(responses)
        self.delay = delay

    def __call__(self, messages, tools=None):
        if not self.responses:
            raise AssertionError("the loop called the model more times than scripted")
        if self.delay:
            time.sleep(self.delay)
        return copy.deepcopy(self.responses.pop(0))


class Dumpable:
    """Stands in for an SDK response object: only .model_dump() is available."""

    def __init__(self, data):
        self._data = data

    def model_dump(self):
        return copy.deepcopy(self._data)


def tool_use(id_, name, input_):
    return {"type": "tool_use", "id": id_, "name": name, "input": input_}


def asks(*blocks, usage=USAGE):
    response = {"id": "msg_x", "type": "message", "role": "assistant", "model": "fake",
                "stop_reason": "tool_use", "content": list(blocks)}
    if usage is not None:
        response["usage"] = usage
    return response


def answers(text, usage=USAGE):
    response = {"id": "msg_y", "type": "message", "role": "assistant", "model": "fake",
                "stop_reason": "end_turn", "content": [{"type": "text", "text": text}]}
    if usage is not None:
        response["usage"] = usage
    return response


def load(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


@pytest.fixture(params=["loop", "mcp"])
def drive(request):
    """Run a scripted model through either loop. The level is the one the run is given."""

    def go(model, level="investigator", max_turns=10, prompt=PROMPT):
        if request.param == "loop":
            from agent import loop
            from agent.permissions import Context

            return loop.run(prompt, model, ctx=Context(level), max_turns=max_turns)
        from agent import mcp_client, mcp_server

        return mcp_client.run_with_mcp(prompt, model, mcp_server.build_server(level), max_turns=max_turns)

    return go


def standard_model(**kwargs):
    """Turn 1: one call that works, one that fails, one the gate refuses (as an investigator).
    Turn 2: the answer."""
    return ScriptedModel(
        asks(
            tool_use("tu_1", "search_logs", {"query": "payment_timeout", "level": "ERROR"}),
            tool_use("tu_2", "read_file", {"path": "shop/does_not_exist_step6.py"}),
            tool_use("tu_3", "write_file", {"path": "shop/app.py", "content": "# new"}),
        ),
        answers("I could not write; a handoff is needed."),
        **kwargs,
    )


def result_blocks(messages):
    found = {}
    for m in messages:
        if m["role"] == "user" and isinstance(m["content"], list):
            for b in m["content"]:
                if b.get("type") == "tool_result":
                    found[b["tool_use_id"]] = b
    return found


# ==========================================================================
# 1. The writer and the reader
# ==========================================================================
def test_write_turn_appends_one_json_line_with_a_timestamp(trace, trace_file):
    trace.write_turn({"run_id": "r1", "turn": 1})
    trace.write_turn({"run_id": "r1", "turn": 2})
    lines = trace_file.read_text().splitlines()
    assert len(lines) == 2, "one line per record, appended"
    first = json.loads(lines[0])
    assert first["run_id"] == "r1" and first["turn"] == 1
    assert datetime.fromisoformat(first["ts"])


def test_write_turn_creates_missing_directories(trace, tmp_path, monkeypatch):
    path = tmp_path / "a" / "b" / "trace.jsonl"
    monkeypatch.setenv("AGENT_TRACE", str(path))
    trace.write_turn({"turn": 1})
    assert path.exists()


def test_write_turn_reads_the_path_at_call_time(trace, tmp_path, monkeypatch):
    one, two = tmp_path / "one.jsonl", tmp_path / "two.jsonl"
    monkeypatch.setenv("AGENT_TRACE", str(one))
    trace.write_turn({"turn": 1})
    monkeypatch.setenv("AGENT_TRACE", str(two))
    trace.write_turn({"turn": 2})
    assert len(load(one)) == 1 and len(load(two)) == 1


def test_records_keep_non_ascii_text_readable_and_valid(trace, trace_file):
    trace.write_turn({"turn": 1, "note": "café ✓ \"quoted\"\nnew line"})
    (record,) = load(trace_file)
    assert record["note"] == "café ✓ \"quoted\"\nnew line"
    assert len(trace_file.read_text().splitlines()) == 1, "a newline inside a value must not split the record"


def test_read_trace_returns_records_in_order_and_skips_blank_lines(trace, tmp_path):
    path = tmp_path / "t.jsonl"
    path.write_text('{"turn": 1}\n\n{"turn": 2}\n   \n')
    assert [r["turn"] for r in trace.read_trace(path)] == [1, 2]


# ==========================================================================
# 2. One record per turn, from either loop
# ==========================================================================
def test_one_record_per_turn(drive, shop_log, trace_file):
    result = drive(standard_model())
    records = load(trace_file)
    assert result["turns"] == 2
    assert [r["turn"] for r in records] == [1, 2], "one record per turn, the answering turn included, 1-based"


def test_records_carry_the_contract_fields(drive, shop_log, trace_file):
    drive(standard_model())
    for r in load(trace_file):
        assert datetime.fromisoformat(r["ts"])
        assert isinstance(r["run_id"], str) and r["run_id"]
        assert isinstance(r["turn"], int) and not isinstance(r["turn"], bool)
        assert r["level"] in ("investigator", "operator")
        assert isinstance(r["stop_reason"], str)
        assert r["usage"] is None or isinstance(r["usage"], dict)
        assert isinstance(r["model_latency_ms"], (int, float)) and r["model_latency_ms"] >= 0
        assert isinstance(r["messages"], list)
        assert isinstance(r["tool_calls"], list)
        for c in r["tool_calls"]:
            assert isinstance(c["id"], str) and isinstance(c["name"], str)
            assert isinstance(c["arguments"], dict)
            assert c["status"] in ("ok", "error", "denied")
            assert isinstance(c["result"], str)
            assert isinstance(c["latency_ms"], (int, float)) and c["latency_ms"] >= 0


def test_run_id_is_shared_within_a_run_and_new_for_the_next(drive, shop_log, trace_file):
    drive(standard_model())
    drive(standard_model())
    records = load(trace_file)
    assert len(records) == 4
    first, second = {r["run_id"] for r in records[:2]}, {r["run_id"] for r in records[2:]}
    assert len(first) == 1 and len(second) == 1, "every turn of one run shares one run_id"
    assert first != second, "a second run must get its own run_id"


@pytest.mark.parametrize("level", ["investigator", "operator"])
def test_level_is_the_level_the_run_used(drive, shop_log, trace_file, level):
    drive(ScriptedModel(answers("nothing to do")), level=level)
    assert [r["level"] for r in load(trace_file)] == [level]


def test_stop_reason_is_the_models_own(drive, shop_log, trace_file):
    drive(standard_model())
    assert [r["stop_reason"] for r in load(trace_file)] == ["tool_use", "end_turn"]


def test_usage_is_copied_from_the_response(drive, shop_log, trace_file):
    rich = {**USAGE, "cache_read_input_tokens": 7}
    drive(ScriptedModel(
        asks(tool_use("tu_1", "search_logs", {"query": "x"}), usage=rich),
        answers("done", usage={"input_tokens": 2000, "output_tokens": 99}),
    ))
    first, second = load(trace_file)
    assert (first["usage"]["input_tokens"], first["usage"]["output_tokens"]) == (1234, 56)
    assert (second["usage"]["input_tokens"], second["usage"]["output_tokens"]) == (2000, 99)


def test_a_response_without_usage_records_none_and_does_not_crash(drive, shop_log, trace_file):
    result = drive(ScriptedModel(answers("hi", usage=None)))
    assert result["stopped"] == "answer"
    assert [r["usage"] for r in load(trace_file)] == [None]


def test_usage_survives_a_response_object_with_model_dump(drive, shop_log, trace_file):
    drive(ScriptedModel(Dumpable(answers("hi"))))
    (record,) = load(trace_file)
    assert record["usage"]["input_tokens"] == 1234 and record["usage"]["output_tokens"] == 56


def test_messages_replay_the_run(drive, shop_log, trace_file):
    result = drive(standard_model())
    records = load(trace_file)
    replayed = [m for r in records for m in r["messages"]]
    assert replayed == result["messages"], (
        "each record holds what was appended during its turn; end to end they must equal the run's messages"
    )
    assert records[0]["messages"][0] == {"role": "user", "content": PROMPT}, "turn 1 starts with the prompt"


def test_tool_calls_describe_each_call_in_order(drive, shop_log, trace_file):
    result = drive(standard_model())
    first, second = load(trace_file)
    assert second["tool_calls"] == [], "the answering turn asked for no tools"
    calls = first["tool_calls"]
    assert [(c["id"], c["name"]) for c in calls] == [
        ("tu_1", "search_logs"), ("tu_2", "read_file"), ("tu_3", "write_file")]
    assert calls[0]["arguments"] == {"query": "payment_timeout", "level": "ERROR"}
    assert calls[2]["arguments"] == {"path": "shop/app.py", "content": "# new"}
    shown_to_model = result_blocks(result["messages"])
    for c in calls:
        assert c["result"] == shown_to_model[c["id"]]["content"], (
            f"{c['name']}: the trace must hold exactly what the model was shown")


def test_status_tells_ok_error_and_denied_apart(drive, shop_log, trace_file):
    drive(standard_model())
    calls = load(trace_file)[0]["tool_calls"]
    assert [(c["name"], c["status"]) for c in calls] == [
        ("search_logs", "ok"), ("read_file", "error"), ("write_file", "denied")]


def test_a_denied_call_keeps_the_reason_the_model_saw(drive, shop_log, trace_file):
    drive(standard_model())
    denied = load(trace_file)[0]["tool_calls"][2]
    assert "permission_denied" in denied["result"]


def test_an_operator_write_is_not_denied(drive, shop_log, trace_file):
    drive(ScriptedModel(
        asks(tool_use("tu_1", "write_file", {"path": "shop/_step6_scratch.txt", "content": "hi"})),
        answers("written"),
    ), level="operator")
    try:
        (call,) = load(trace_file)[0]["tool_calls"]
        assert call["status"] == "ok"
    finally:
        (REPO / "shop" / "_step6_scratch.txt").unlink(missing_ok=True)


def test_a_turn_without_tool_calls_has_an_empty_list(drive, shop_log, trace_file):
    drive(ScriptedModel(answers("nothing needed")))
    (record,) = load(trace_file)
    assert record["tool_calls"] == []


def test_a_run_cut_off_by_max_turns_still_traces_every_turn(drive, shop_log, trace_file):
    looping = [asks(tool_use(f"tu_{i}", "search_logs", {"query": "x"})) for i in range(3)]
    result = drive(ScriptedModel(*looping), max_turns=3)
    assert result["stopped"] == "max_turns"
    records = load(trace_file)
    assert [r["turn"] for r in records] == [1, 2, 3]
    assert [m for r in records for m in r["messages"]] == result["messages"]


def test_the_trace_and_the_audit_log_stay_separate_files(drive, shop_log, trace_file, audit_log):
    drive(standard_model())
    audit = load(audit_log)
    assert sum(a["event"] == "permission_denied" for a in audit) == 1
    assert all("turn" not in a and "run_id" not in a for a in audit), "turn records belong in the trace file"
    assert len(load(trace_file)) == 2


# ==========================================================================
# 3. Latency
# ==========================================================================
def test_model_latency_is_the_model_call_in_milliseconds(drive, shop_log, trace_file):
    drive(ScriptedModel(answers("slow thinker"), delay=0.15))
    (record,) = load(trace_file)
    assert 100 <= record["model_latency_ms"] < 5000, (
        f"a 150 ms model call came out as {record['model_latency_ms']}: milliseconds, around the call only")


def test_tool_latency_times_the_tool_not_the_model(trace_file):
    from agent import loop

    def sleepy(**_):
        time.sleep(0.15)
        return {"ok": True}

    model = ScriptedModel(asks(tool_use("tu_1", "sleepy", {})), answers("done"))
    loop.run(PROMPT, model, tools={"sleepy": sleepy})
    first, _ = load(trace_file)
    assert first["tool_calls"][0]["latency_ms"] >= 100
    assert first["model_latency_ms"] < 100, "the tool's time leaked into the model's"


def test_a_denied_call_never_runs_the_tool(trace_file):
    from agent import loop

    ran = []

    def slow_write(**_):
        ran.append(1)
        time.sleep(0.15)
        return {"ok": True}

    model = ScriptedModel(
        asks(tool_use("tu_1", "write_file", {"path": "shop/app.py", "content": "x"})), answers("blocked"))
    loop.run(PROMPT, model, tools={"write_file": slow_write})
    first, _ = load(trace_file)
    (call,) = first["tool_calls"]
    assert call["status"] == "denied" and not ran
    assert call["latency_ms"] < 100, "a refused call has nothing to time"


def test_latency_is_never_negative_when_the_wall_clock_jumps(trace_file, monkeypatch):
    """Durations come from a clock that only moves forward. The wall clock can be stepped
    backwards (NTP, a manual change), and then an end-minus-start subtraction goes negative."""
    from agent import loop

    state = {"t": 1_700_000_000.0}

    def jumping_time():
        state["t"] -= 1000.0
        return state["t"]

    def echo(**kw):
        return kw

    model = ScriptedModel(asks(tool_use("tu_1", "echo", {"a": 1})), answers("done"))
    with monkeypatch.context() as m:
        m.setattr(time, "time", jumping_time)
        m.setattr(time, "time_ns", lambda: int(jumping_time() * 1e9))
        loop.run(PROMPT, model, tools={"echo": echo})
    for record in load(trace_file):
        assert record["model_latency_ms"] >= 0, "model latency went negative: it was timed with the wall clock"
        for call in record["tool_calls"]:
            assert call["latency_ms"] >= 0, "tool latency went negative: it was timed with the wall clock"


# ==========================================================================
# 4. Carry-forward: create_handoff errors reach the model with their reason
# ==========================================================================
@pytest.mark.parametrize("arguments, word", [
    ({"summary": "s", "evidence": [], "requested_action": "a"}, "evidence"),
    ({"summary": "   ", "evidence": ["e"], "requested_action": "a"}, "summary"),
    ({"summary": "s", "evidence": ["e"], "requested_action": ""}, "requested"),
])
def test_create_handoff_validation_errors_name_the_problem(arguments, word, trace_file, audit_log):
    from agent import mcp_client, mcp_server

    model = ScriptedModel(asks(tool_use("tu_1", "create_handoff", arguments)), answers("ok"))
    result = mcp_client.run_with_mcp(PROMPT, model, mcp_server.build_server("investigator"))
    (block,) = result_blocks(result["messages"]).values()
    assert block["is_error"] is True
    text = block["content"]
    assert text.strip() != "Error executing tool create_handoff", "the model is only told the tool failed, not why"
    assert word in text.lower(), f"the error should name the problem ({word!r}), got {text!r}"
    assert [c["status"] for c in load(trace_file)[0]["tool_calls"]] == ["error"]
    assert not audit_log.exists() or not any(a["event"] == "handoff" for a in load(audit_log))


# ==========================================================================
# 5. The viewer
# ==========================================================================
def excerpt(denied=True, usage=True):
    """A run as someone would paste it: only fields from the contract, nothing from a loop."""
    denial = json.dumps({"error": "permission_denied", "tool": "write_file", "level": "investigator",
                         "reason": "investigator does not have permission to run tool write_file.",
                         "suggestion": "Use tool create_handoff to ask someone with a higher level to do this."})
    t1_calls = [{
        "id": "tu_1", "name": "write_file",
        "arguments": {"path": "shop/app.py", "content": "NEW-CONTENT-MARKER"},
        "status": "denied" if denied else "ok",
        "result": denial if denied else "WRITE-OK-MARKER",
        "latency_ms": 0.4,
    }]
    return [
        {"ts": "2026-11-11T21:54:40.100+00:00", "run_id": "run-A", "turn": 1, "level": "investigator",
         "stop_reason": "tool_use", "usage": {"input_tokens": 4321, "output_tokens": 8765} if usage else None,
         "model_latency_ms": 812.5,
         "messages": [
             {"role": "user", "content": "PROMPT-MARKER fix the checkout bug"},
             {"role": "assistant", "content": [
                 {"type": "text", "text": "I will edit the file."},
                 {"type": "tool_use", "id": "tu_1", "name": "write_file",
                  "input": {"path": "shop/app.py", "content": "NEW-CONTENT-MARKER"}}]},
             {"role": "user", "content": [
                 {"type": "tool_result", "tool_use_id": "tu_1", "is_error": denied,
                  "content": t1_calls[0]["result"]}]}],
         "tool_calls": t1_calls},
        {"ts": "2026-11-11T21:54:42.900+00:00", "run_id": "run-A", "turn": 2, "level": "investigator",
         "stop_reason": "end_turn", "usage": {"input_tokens": 5000, "output_tokens": 61} if usage else None,
         "model_latency_ms": 640.0,
         "messages": [{"role": "assistant", "content": [{"type": "text", "text": "ANSWER-MARKER all done"}]}],
         "tool_calls": []},
    ]


def test_render_run_shows_the_conversation(view):
    text = view.render_run(excerpt())
    assert isinstance(text, str)
    for marker in ("PROMPT-MARKER", "I will edit the file.", "write_file", "NEW-CONTENT-MARKER", "ANSWER-MARKER"):
        assert marker in text, f"the viewer should show {marker!r}"
    assert "4321" in text and "8765" in text, "token counts per turn belong in the view"


def test_render_run_keeps_the_order_of_the_run(view):
    text = view.render_run(excerpt())
    assert text.index("PROMPT-MARKER") < text.index("write_file") < text.index("ANSWER-MARKER")


def test_render_run_makes_a_denied_call_easy_to_spot(view):
    assert "denied" in view.render_run(excerpt(denied=True)).lower()


def test_render_run_does_not_cry_wolf_on_a_clean_run(view):
    text = view.render_run(excerpt(denied=False)).lower()
    assert "denied" not in text and "WRITE-OK-MARKER".lower() in text


def test_render_run_copes_with_missing_usage(view):
    assert "ANSWER-MARKER" in view.render_run(excerpt(usage=False))


def test_the_viewer_reads_what_a_real_run_leaves_behind(drive, shop_log, trace_file, trace, view):
    drive(standard_model())
    text = view.render_run(trace.read_trace(trace_file))
    assert "search_logs" in text and "write_file" in text
    assert "denied" in text.lower()
    assert "1234" in text


def two_run_file(path):
    first, second = excerpt(), excerpt(denied=False)
    for r in first:
        r["run_id"] = "run-A"
        r["messages"] = json.loads(json.dumps(r["messages"]).replace("PROMPT-MARKER", "FIRST-RUN-PROMPT"))
    for r in second:
        r["run_id"] = "run-B"
        r["messages"] = json.loads(json.dumps(r["messages"]).replace("PROMPT-MARKER", "SECOND-RUN-PROMPT"))
    path.write_text("".join(json.dumps(r) + "\n" for r in first + second))


def cli(*args, env=None):
    return subprocess.run([sys.executable, "-m", "agent.view_trace", *args], cwd=REPO, capture_output=True,
                          text=True, timeout=60, env={**os.environ, **(env or {})})


def test_cli_prints_the_last_run_by_default(tmp_path):
    path = tmp_path / "two.jsonl"
    two_run_file(path)
    out = cli(str(path))
    assert out.returncode == 0, out.stderr
    assert "SECOND-RUN-PROMPT" in out.stdout and "FIRST-RUN-PROMPT" not in out.stdout


def test_cli_run_flag_picks_a_run(tmp_path):
    path = tmp_path / "two.jsonl"
    two_run_file(path)
    out = cli(str(path), "--run", "run-A")
    assert out.returncode == 0, out.stderr
    assert "FIRST-RUN-PROMPT" in out.stdout and "SECOND-RUN-PROMPT" not in out.stdout


def test_cli_defaults_to_the_agent_trace_file(tmp_path):
    path = tmp_path / "two.jsonl"
    two_run_file(path)
    out = cli(env={"AGENT_TRACE": str(path)})
    assert out.returncode == 0, out.stderr
    assert "SECOND-RUN-PROMPT" in out.stdout


def test_cli_unknown_run_exits_non_zero_without_a_traceback(tmp_path):
    path = tmp_path / "two.jsonl"
    two_run_file(path)
    out = cli(str(path), "--run", "run-nope")
    assert out.returncode != 0 and "Traceback" not in out.stderr
    assert "run-nope" in out.stderr + out.stdout, "say which run id was not found"
    assert "No module named" not in out.stderr, "agent/view_trace.py doesn't exist yet"


def test_cli_missing_file_exits_non_zero_without_a_traceback(tmp_path):
    out = cli(str(tmp_path / "nope.jsonl"))
    assert out.returncode != 0 and "Traceback" not in out.stderr
    assert "nope.jsonl" in out.stderr + out.stdout, "say which file could not be read"
    assert "No module named" not in out.stderr, "agent/view_trace.py doesn't exist yet"

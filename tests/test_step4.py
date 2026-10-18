"""Step 4 acceptance tests, part 1: permissions as code. No API calls.

Part 2 (evidence from one real run) is tests/test_step4_artifacts.py.
Don't edit these files; if a test looks wrong, raise it and we'll change it together.

What these tests pin (the contract), and nothing more:

agent/permissions.py  (new)
    Context(level: str = "investigator")
        - A small, frozen dataclass the loop carries. Two levels: "investigator"
          (read-only, the default) and "operator" (may write). Any other level
          raises ValueError when the Context is created.
    check(ctx: Context, name: str, arguments: dict) -> dict | None
        - The gate. Returns None when the call is allowed, otherwise a denial dict:
            {"error": "permission_denied", "tool": name, "level": ctx.level,
             "reason": <non-empty str>, "suggestion": <non-empty str>}
        - Rules:
            write_file needs "operator".
            Any call with a "path" argument must resolve inside shop/, at every level.
            Everything else (search_logs, run_query, read_file inside shop/,
            create_handoff) is allowed at every level.
        - When an investigator is denied a write, the suggestion names create_handoff,
          so the model knows how to escalate instead of retrying.
        - It decides; it doesn't run the tool.

agent/tools.py  (adds to Step 3)
    write_file(path: str, content: str) -> dict
        - Writes text to a file inside shop/ (same path rule as read_file, relative to
          the repo root) and returns a JSON-serialisable dict. Raises PermissionError
          outside shop/, before writing anything. This is defence in depth: the gate
          should already have stopped it.
    create_handoff(summary: str, evidence: list[str], requested_action: str) -> dict
        - Builds a structured handoff record, appends it to the audit log (below) and
          returns it: {"id": <non-empty str, unique per call>, "summary": ...,
          "evidence": [...], "requested_action": ..., "status": "open"}.
        - Empty summary, empty requested_action, or evidence that isn't a non-empty
          list of strings raises ValueError. A handoff without evidence is a guess.
    WRITE_FILE_PARAMETERS, HANDOFF_PARAMETERS: dict     (JSON Schema, like Step 3's)
    TOOLS now also holds "write_file" and "create_handoff".

agent/claude_raw.py
    tool_definitions() covers all five tools (the Step 3 test already checks it).

The audit log
    - JSON lines at AGENT_LOG (read at call time; default logs/agent.log, which is
      gitignored). Every line has "ts" and "event".
    - Each denied call writes exactly one line:
        {"event": "permission_denied", "level", "tool", "arguments", "reason", ...}
    - Each handoff writes exactly one line:
        {"event": "handoff", "id", "summary", "evidence", "requested_action", ...}

agent/loop.py
    run(prompt, call_model, tools=None, max_turns=10, ctx=None) -> dict
        - ctx defaults to Context(), so a run is read-only unless you say otherwise.
        - Every tool call goes through check() before it runs. A denied call never
          runs the tool: it becomes a tool_result with "is_error": True whose content
          is json.dumps(denial), and it writes one audit line.
        - Every tool_result in a turn (success, error or denial) carries its
          tool_use_id, and they all go back in ONE user message, in call order.
          (This closes a gap in the Step 3 tests: an error result must still answer
          its tool_use, or the real API rejects the request.)

How you structure the code beyond these names is up to you. The Step 3 size test still
applies to loop.py: the gate belongs in permissions.py, not inline in the loop.
"""
import copy
import importlib
import json
import uuid
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SHOP_DIR = REPO / "shop"


@pytest.fixture(autouse=True)
def audit_log(tmp_path, monkeypatch):
    """Every test gets its own audit log, so nothing lands in the real logs/."""
    path = tmp_path / "agent.log"
    monkeypatch.setenv("AGENT_LOG", str(path))
    return path


@pytest.fixture
def perms():
    return importlib.import_module("agent.permissions")


@pytest.fixture
def tools():
    return importlib.import_module("agent.tools")


@pytest.fixture
def claude():
    return importlib.import_module("agent.claude_raw")


@pytest.fixture
def loop():
    return importlib.import_module("agent.loop")


@pytest.fixture
def scratch():
    """A throwaway file name inside shop/, removed afterwards."""
    rel = f"shop/_step4_scratch_{uuid.uuid4().hex[:8]}.txt"
    yield rel
    (REPO / rel).unlink(missing_ok=True)


def audit_lines(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def events(path, kind):
    return [r for r in audit_lines(path) if r.get("event") == kind]


# --------------------------------------------------------------------------
# The same scripted stand-in for the model as Step 3.
# --------------------------------------------------------------------------
class ScriptedModel:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, messages):
        self.calls.append(copy.deepcopy(messages))
        if not self.responses:
            raise AssertionError("the loop called the model more times than scripted")
        return self.responses.pop(0)


def tool_use(id_, name, input_):
    return {"type": "tool_use", "id": id_, "name": name, "input": input_}


def asks(*blocks):
    return {"id": "msg_x", "type": "message", "role": "assistant", "model": "fake",
            "stop_reason": "tool_use", "content": list(blocks)}


def answers(text):
    return {"id": "msg_y", "type": "message", "role": "assistant", "model": "fake",
            "stop_reason": "end_turn", "content": [{"type": "text", "text": text}]}


def results_in(message):
    assert message["role"] == "user"
    return [b for b in message["content"] if b.get("type") == "tool_result"]


class Spy:
    """A tool that records whether it ran."""

    def __init__(self, returns="ran"):
        self.calls = []
        self.returns = returns

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return self.returns


# ==========================================================================
# 1. Context
# ==========================================================================
def test_context_defaults_to_investigator(perms):
    assert perms.Context().level == "investigator"


def test_context_accepts_operator(perms):
    assert perms.Context("operator").level == "operator"


@pytest.mark.parametrize("level", ["admin", "root", "", "Operator "])
def test_context_rejects_unknown_levels(perms, level):
    with pytest.raises(ValueError):
        perms.Context(level)


def test_context_is_frozen(perms):
    ctx = perms.Context()
    with pytest.raises(Exception):
        ctx.level = "operator"


# ==========================================================================
# 2. The gate
# ==========================================================================
@pytest.mark.parametrize("level", ["investigator", "operator"])
@pytest.mark.parametrize("name,arguments", [
    ("search_logs", {"query": "timeout"}),
    ("run_query", {"sql": "SELECT 1"}),
    ("read_file", {"path": "shop/app.py"}),
    ("create_handoff", {"summary": "s", "evidence": ["e"], "requested_action": "a"}),
])
def test_gate_allows_reads_and_handoffs_at_every_level(perms, level, name, arguments):
    assert perms.check(perms.Context(level), name, arguments) is None


def test_gate_allows_an_operator_to_write_inside_shop(perms):
    assert perms.check(perms.Context("operator"), "write_file",
                       {"path": "shop/app.py", "content": "x"}) is None


def test_gate_denies_an_investigator_write(perms):
    denial = perms.check(perms.Context("investigator"), "write_file",
                         {"path": "shop/app.py", "content": "x"})
    assert denial is not None
    assert denial["error"] == "permission_denied"
    assert denial["tool"] == "write_file"
    assert denial["level"] == "investigator"
    assert denial["reason"].strip()
    assert "create_handoff" in denial["suggestion"], "tell the model how to escalate"


@pytest.mark.parametrize("level", ["investigator", "operator"])
@pytest.mark.parametrize("name", ["read_file", "write_file"])
@pytest.mark.parametrize("path", [
    ".env",
    "shop/../.env",
    "agent/tools.py",
    "/etc/passwd",
    str(SHOP_DIR) + "_evil/x.py",
])
def test_gate_denies_paths_outside_shop_at_every_level(perms, level, name, path):
    denial = perms.check(perms.Context(level), name, {"path": path, "content": "x"})
    assert denial is not None, f"{level} {name}({path!r}) should be denied"
    assert denial["error"] == "permission_denied"
    assert denial["tool"] == name and denial["level"] == level
    assert denial["reason"].strip() and denial["suggestion"].strip()


def test_denial_is_json_serialisable(perms):
    denial = perms.check(perms.Context(), "write_file", {"path": "shop/app.py", "content": "x"})
    assert json.loads(json.dumps(denial)) == denial


# ==========================================================================
# 3. The two new tools
# ==========================================================================
def test_write_file_writes_inside_shop(tools, scratch):
    out = tools.write_file(scratch, "hello")
    assert (REPO / scratch).read_text() == "hello"
    json.dumps(out)


def test_write_file_works_from_any_working_directory(tools, scratch, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tools.write_file(scratch, "hi")
    assert (REPO / scratch).read_text() == "hi"


@pytest.mark.parametrize("path", [".env.step4", "shop/../step4_escape.txt", str(SHOP_DIR) + "_evil/x.txt"])
def test_write_file_refuses_outside_shop_on_its_own(tools, path):
    target = (REPO / path).resolve() if not Path(path).is_absolute() else Path(path)
    with pytest.raises(PermissionError):
        tools.write_file(path, "nope")
    assert not target.exists(), "refuse before writing anything"


def test_create_handoff_returns_a_structured_record(tools):
    record = tools.create_handoff("Stock missing on search", ["shop/app.py: search() has no stock"], "Add stock to search results")
    assert record["summary"] == "Stock missing on search"
    assert record["evidence"] == ["shop/app.py: search() has no stock"]
    assert record["requested_action"] == "Add stock to search results"
    assert record["status"] == "open"
    assert isinstance(record["id"], str) and record["id"].strip()
    json.dumps(record)


def test_create_handoff_ids_are_unique(tools):
    a = tools.create_handoff("s", ["e"], "a")
    b = tools.create_handoff("s", ["e"], "a")
    assert a["id"] != b["id"]


def test_create_handoff_writes_one_audit_line(tools, audit_log):
    record = tools.create_handoff("s", ["e1", "e2"], "do the thing")
    [line] = events(audit_log, "handoff")
    assert line["id"] == record["id"]
    assert line["evidence"] == ["e1", "e2"]
    assert line["requested_action"] == "do the thing"
    assert "ts" in line


def test_audit_log_path_is_read_at_call_time(tools, tmp_path, monkeypatch):
    other = tmp_path / "elsewhere" / "agent.log"
    monkeypatch.setenv("AGENT_LOG", str(other))
    tools.create_handoff("s", ["e"], "a")
    assert len(events(other, "handoff")) == 1


@pytest.mark.parametrize("summary,evidence,action", [
    ("", ["e"], "a"),
    ("   ", ["e"], "a"),
    ("s", [], "a"),
    ("s", "just a string", "a"),
    ("s", ["e"], ""),
])
def test_create_handoff_rejects_empty_or_unsupported_input(tools, audit_log, summary, evidence, action):
    with pytest.raises(ValueError):
        tools.create_handoff(summary, evidence, action)
    assert events(audit_log, "handoff") == [], "don't log a handoff you rejected"


@pytest.mark.parametrize("name,required", [
    ("WRITE_FILE_PARAMETERS", ["path", "content"]),
    ("HANDOFF_PARAMETERS", ["summary", "evidence", "requested_action"]),
])
def test_new_schemas_describe_their_arguments(tools, name, required):
    schema = getattr(tools, name)
    assert schema["type"] == "object"
    assert sorted(schema.get("required", [])) == sorted(required)
    for prop_name, prop in schema["properties"].items():
        assert prop.get("description"), f"{prop_name} needs a description: the model reads it"


def test_handoff_schema_asks_for_a_list_of_evidence(tools):
    evidence = tools.HANDOFF_PARAMETERS["properties"]["evidence"]
    assert evidence["type"] == "array"
    assert evidence["items"]["type"] == "string"


def test_registry_holds_all_five_tools(tools, claude):
    assert set(tools.TOOLS) == {"search_logs", "read_file", "run_query", "write_file", "create_handoff"}
    assert tools.TOOLS["write_file"] is tools.write_file
    assert tools.TOOLS["create_handoff"] is tools.create_handoff
    defs = {d["name"]: d for d in claude.tool_definitions()}
    assert defs["write_file"]["input_schema"] == tools.WRITE_FILE_PARAMETERS
    assert defs["create_handoff"]["input_schema"] == tools.HANDOFF_PARAMETERS


# ==========================================================================
# 4. The gate inside the loop
# ==========================================================================
def test_loop_is_read_only_by_default(loop, scratch, audit_log):
    model = ScriptedModel(asks(tool_use("t1", "write_file", {"path": scratch, "content": "x"})), answers("ok"))
    out = loop.run("change it", model)
    assert out["stopped"] == "answer"
    assert not (REPO / scratch).exists(), "no ctx means investigator: the write must not happen"
    [result] = results_in(model.calls[1][-1])
    assert result["tool_use_id"] == "t1"
    assert result["is_error"] is True
    assert json.loads(result["content"])["error"] == "permission_denied"


def test_a_denied_call_never_runs_the_tool(loop, perms):
    spy = Spy()
    model = ScriptedModel(asks(tool_use("t1", "write_file", {"path": "shop/app.py", "content": "x"})), answers("ok"))
    loop.run("go", model, tools={"write_file": spy}, ctx=perms.Context("investigator"))
    assert spy.calls == [], "the gate runs before the tool, not after"


def test_denial_goes_back_to_the_model_exactly_as_the_gate_built_it(loop, perms):
    ctx = perms.Context("investigator")
    args = {"path": "shop/app.py", "content": "x"}
    model = ScriptedModel(asks(tool_use("t1", "write_file", args)), answers("ok"))
    loop.run("go", model, tools={"write_file": Spy()}, ctx=ctx)
    [result] = results_in(model.calls[1][-1])
    assert json.loads(result["content"]) == perms.check(ctx, "write_file", args)


def test_a_denial_writes_exactly_one_audit_line(loop, perms, audit_log):
    args = {"path": ".env"}
    model = ScriptedModel(asks(tool_use("t1", "read_file", args)), answers("ok"))
    loop.run("read the env", model, tools={"read_file": Spy()}, ctx=perms.Context("operator"))
    [line] = events(audit_log, "permission_denied")
    assert line["tool"] == "read_file"
    assert line["level"] == "operator"
    assert line["arguments"] == args
    assert line["reason"].strip()
    assert "ts" in line


def test_allowed_calls_write_no_denial_lines(loop, perms, audit_log):
    model = ScriptedModel(asks(tool_use("t1", "read_file", {"path": "shop/app.py"})), answers("ok"))
    loop.run("go", model, tools={"read_file": Spy()}, ctx=perms.Context())
    assert events(audit_log, "permission_denied") == []


def test_an_operator_write_goes_through(loop, perms, scratch):
    model = ScriptedModel(asks(tool_use("t1", "write_file", {"path": scratch, "content": "done"})), answers("ok"))
    loop.run("write it", model, ctx=perms.Context("operator"))
    assert (REPO / scratch).read_text() == "done"
    [result] = results_in(model.calls[1][-1])
    assert result["tool_use_id"] == "t1" and not result.get("is_error")


def test_denied_write_then_handoff(loop, perms, audit_log, scratch):
    """The Step 7 scenario D shape: try to write, get denied, escalate."""
    model = ScriptedModel(
        asks(tool_use("t1", "write_file", {"path": scratch, "content": "x"})),
        asks(tool_use("t2", "create_handoff", {
            "summary": "Need a write", "evidence": ["denied write_file"], "requested_action": "Apply the change"})),
        answers("Escalated."),
    )
    out = loop.run("change it", model, ctx=perms.Context())
    assert out["stopped"] == "answer"
    assert not (REPO / scratch).exists()
    [handoff_result] = results_in(model.calls[2][-1])
    assert handoff_result["tool_use_id"] == "t2" and not handoff_result.get("is_error")
    record = json.loads(handoff_result["content"])
    assert record["status"] == "open"
    assert [e["event"] for e in audit_lines(audit_log)] == ["permission_denied", "handoff"]


def test_every_result_in_a_turn_answers_its_call_in_one_message(loop, perms):
    """Allowed, throwing, denied and unknown calls in one turn: one user message, ids in order."""
    def broken(**_):
        raise RuntimeError("disk on fire")

    model = ScriptedModel(
        asks(
            tool_use("a", "echo", {"text": "1"}),
            tool_use("b", "broken", {}),
            tool_use("c", "write_file", {"path": "shop/app.py", "content": "x"}),
            tool_use("d", "rm_rf", {}),
        ),
        answers("ok"),
    )
    loop.run("go", model, tools={"echo": lambda text: text, "broken": broken, "write_file": Spy()},
             ctx=perms.Context())
    history = model.calls[1]
    assert [m["role"] for m in history] == ["user", "assistant", "user"], (
        "all results for a turn, errors included, belong in a single user message"
    )
    results = results_in(history[-1])
    assert [r.get("tool_use_id") for r in results] == ["a", "b", "c", "d"], (
        "every tool_result needs the tool_use_id it answers, or the real API rejects the request"
    )
    assert [bool(r.get("is_error")) for r in results] == [False, True, True, True]
    assert json.loads(results[2]["content"])["error"] == "permission_denied"

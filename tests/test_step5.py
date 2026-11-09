"""Step 5 acceptance tests, part 1: a minimal MCP server, and the loop as its client. No API calls.

Part 2 (evidence from the Inspector and one real run) is tests/test_step5_artifacts.py.
Don't edit these files; if a test looks wrong, raise it and we'll change it together.

These tests talk to your server the way a real client does: through the MCP Python SDK's
`Client`, either in-process (passing your MCPServer object) or over stdio (launching
`python -m agent.mcp_server`). Nothing here imports your tool functions to call them
directly, because in Step 5 nothing outside your server should.

What these tests pin (the contract), and nothing more:

pyproject.toml
    The `agent` extra pins the MCP SDK to the 2.x line with an exact or compatible
    pin (for example "mcp==2.3.0" or "mcp~=2.3"). Pre-2.0 tutorials are out of date.

agent/mcp_server.py  (new)
    build_server(level: str = "investigator") -> mcp.server.mcpserver.MCPServer
        - The permission level is fixed when the server is built. An MCP request carries
          no Context of yours, so the server has to know its level on its own.
          An unknown level raises ValueError.
        - Exposes at least: search_logs, read_file, write_file, create_handoff.
          (write_file is exposed so the Step 4 scenario can be rerun: an investigator
          server must deny it by itself.) run_query is optional.
        - The schema each tool advertises keeps what your Step 4 schemas promised:
          the same required arguments, and search_logs' level is still limited to
          INFO, WARN and ERROR. The SDK builds schemas from your type hints, so check
          what it actually advertises; the Step 2 "warning" bug can come back here.
        - Every call is gated by the server, with the same rules as Step 4's check():
          scope first (any path outside shop/ is refused at every level, with no
          handoff suggestion), then level (an investigator's write_file is denied
          with a suggestion naming create_handoff).
        - A denied call never runs the tool, comes back with isError true, writes exactly
          one permission_denied line to the audit log (AGENT_LOG, with the arguments),
          and its text reaches the client: "permission_denied", the reason and the
          suggestion. The model-facing text does NOT echo the call's arguments.
        - A tool that fails for an ordinary reason (a missing file inside shop/) comes
          back with isError true and a message the model can act on, not just a
          generic "Error executing tool".
    `python -m agent.mcp_server` runs the server over stdio, at the level in the
    AGENT_LEVEL environment variable (default "investigator"). This is how the
    Inspector and other clients launch it.

agent/mcp_client.py  (new)
    tool_definitions_from_mcp(list_tools_result) -> list[dict]
        - Turns the SDK's ListToolsResult into Claude tool definitions:
          [{"name", "description", "input_schema"}, ...], one per server tool.
          This is the hop a tool definition takes from the server to the model.
    to_tool_result(tool_use_id: str, call_tool_result) -> dict
        - Turns the SDK's CallToolResult into one Claude tool_result block:
          {"type": "tool_result", "tool_use_id": ..., "content": <str>, "is_error": <bool>}
          where content holds the result's text. is_error mirrors the server's isError.
    run_with_mcp(prompt, call_model, server, max_turns=10) -> dict
        - The Step 4 loop, with the tools living behind an MCP server. `server` is
          anything mcp.Client accepts (an MCPServer object in these tests).
        - call_model is called as call_model(messages, tools), where tools is the list
          from tool_definitions_from_mcp. The client discovers the tools; you don't
          hand-write them a second time.
        - Same return shape as loop.run: {"answer", "messages", "turns", "stopped"},
          "stopped" is "answer" or "max_turns", all results of a turn go back in one
          user message in call order.
        - It has no ctx: the level is the server's. It must not add a second gate with a
          level of its own (an operator server's write goes through), and a denial is
          logged once, by the server.
        - It's a plain (synchronous) function; how you bridge to the async SDK is up to you.

agent/permissions.py and agent/loop.py  (carry-forwards from the Step 4 review)
    - check() refuses an out-of-scope path before looking at the level, and that refusal
      doesn't suggest a handoff, even for a write by an investigator.
    - The denial dict the model sees has no "arguments" key; the audit line keeps them.
    - A non-string path is denied, not a crash.
    - An answer on the last allowed turn is an "answer", not "max_turns".

How you structure the code beyond these names is up to you.
"""
import copy
import importlib
import json
import os
import re
import subprocess
import sys
import tomllib
import uuid
from pathlib import Path

import anyio
import pytest

REPO = Path(__file__).resolve().parents[1]
SENTINEL = "SENTINEL-CONTENT-7f3a"


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def audit_log(tmp_path, monkeypatch):
    path = tmp_path / "agent.log"
    monkeypatch.setenv("AGENT_LOG", str(path))
    return path


@pytest.fixture
def shop_log(tmp_path, monkeypatch):
    path = tmp_path / "app.log"
    lines = [
        {"ts": 1, "level": "INFO", "event": "search", "query": "mug", "duration_ms": 12},
        {"ts": 2, "level": "ERROR", "event": "payment_timeout", "order_total": 142.5},
        {"ts": 3, "level": "WARN", "event": "out_of_stock", "sku": "MUG-1"},
    ]
    path.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    monkeypatch.setenv("SHOP_LOG", str(path))
    return path


@pytest.fixture
def scratch():
    rel = f"shop/_step5_scratch_{uuid.uuid4().hex[:8]}.txt"
    yield rel
    (REPO / rel).unlink(missing_ok=True)


@pytest.fixture
def mcp():
    return importlib.import_module("mcp")


@pytest.fixture
def srv():
    return importlib.import_module("agent.mcp_server")


@pytest.fixture
def cli():
    return importlib.import_module("agent.mcp_client")


@pytest.fixture
def perms():
    return importlib.import_module("agent.permissions")


@pytest.fixture
def loop():
    return importlib.import_module("agent.loop")


def audit_lines(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def events(path, kind):
    return [r for r in audit_lines(path) if r.get("event") == kind]


def call(server, name, arguments):
    """Call one tool through a real MCP client, in-process."""
    from mcp import Client

    async def go():
        async with Client(server) as client:
            return await client.call_tool(name, arguments)

    return anyio.run(go)


def list_tools(server):
    from mcp import Client

    async def go():
        async with Client(server) as client:
            return await client.list_tools()

    return anyio.run(go)


def text_of(result):
    return "\n".join(getattr(c, "text", "") for c in result.content)


def schemas(server):
    return {t.name: t.input_schema for t in list_tools(server).tools}


def enum_values(schema):
    """All enum values anywhere in a (possibly anyOf/$ref-wrapped) JSON schema."""
    found = []

    def walk(node):
        if isinstance(node, dict):
            if "enum" in node:
                found.extend(node["enum"])
            if "const" in node:
                found.append(node["const"])
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(schema)
    return set(found)


# Scripted model for run_with_mcp: called as call_model(messages, tools).
class ScriptedModel:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []
        self.tools_seen = []

    def __call__(self, messages, tools):
        self.calls.append(copy.deepcopy(messages))
        self.tools_seen.append(copy.deepcopy(tools))
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


# ==========================================================================
# 1. The SDK pin
# ==========================================================================
def test_agent_extra_pins_mcp_2():
    extras = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]["optional-dependencies"]["agent"]
    pins = [d for d in extras if re.match(r"^mcp\b", d.replace(" ", ""))]
    assert pins, "add the MCP SDK to the agent extra in pyproject.toml"
    assert re.match(r"^mcp(\[[^\]]*\])?(==|~=)2\.", pins[0].replace(" ", "")), (
        f"pin the 2.x SDK exactly or compatibly (==2.x.y or ~=2.x), got {pins[0]!r}"
    )


def test_installed_sdk_is_2x(mcp):
    from importlib.metadata import version

    assert version("mcp").split(".")[0] == "2"


# ==========================================================================
# 2. The server: what it advertises
# ==========================================================================
def test_build_server_returns_an_mcpserver(srv):
    from mcp.server.mcpserver import MCPServer

    assert isinstance(srv.build_server(), MCPServer)
    assert isinstance(srv.build_server("operator"), MCPServer)


def test_build_server_rejects_unknown_levels(srv):
    with pytest.raises(ValueError):
        srv.build_server("admin")


def test_server_lists_the_tools(srv):
    names = {t.name for t in list_tools(srv.build_server()).tools}
    assert {"search_logs", "read_file", "write_file", "create_handoff"} <= names


def test_every_tool_has_a_description(srv):
    for t in list_tools(srv.build_server()).tools:
        assert t.description and t.description.strip(), f"{t.name} has no description for the model"


@pytest.mark.parametrize("name, required", [
    ("search_logs", {"query"}),
    ("read_file", {"path"}),
    ("write_file", {"path", "content"}),
    ("create_handoff", {"summary", "evidence", "requested_action"}),
])
def test_advertised_schemas_keep_their_required_arguments(srv, name, required):
    schema = schemas(srv.build_server())[name]
    assert set(schema.get("required", [])) == required


def test_search_logs_level_is_still_an_enum(srv):
    schema = schemas(srv.build_server())["search_logs"]
    level = schema["properties"]["level"]
    assert {"INFO", "WARN", "ERROR"} <= enum_values({"level": level, "$defs": schema.get("$defs", {})}), (
        "search_logs' level lost its enum on the way through MCP: look at what the SDK builds from your type hints"
    )


def test_handoff_evidence_is_still_a_list_of_strings(srv):
    evidence = schemas(srv.build_server())["create_handoff"]["properties"]["evidence"]
    flat = json.dumps(evidence)
    assert '"array"' in flat and '"string"' in flat


# ==========================================================================
# 3. The server: calls that are allowed
# ==========================================================================
def test_search_logs_through_mcp(srv, shop_log):
    result = call(srv.build_server(), "search_logs", {"query": "payment", "level": "ERROR"})
    assert not result.is_error
    assert "payment_timeout" in text_of(result) or "payment_timeout" in json.dumps(result.structured_content or {})
    assert "out_of_stock" not in text_of(result)


def test_read_file_inside_shop_through_mcp(srv):
    result = call(srv.build_server(), "read_file", {"path": "shop/__init__.py"})
    assert not result.is_error
    assert text_of(result) == (REPO / "shop" / "__init__.py").read_text() or (REPO / "shop" / "__init__.py").read_text() in text_of(result)


def test_create_handoff_through_mcp(srv, audit_log):
    result = call(srv.build_server(), "create_handoff", {
        "summary": "Show stock on search results",
        "evidence": ["shop/app.py builds search results"],
        "requested_action": "write_file shop/app.py",
    })
    assert not result.is_error
    handoffs = events(audit_log, "handoff")
    assert len(handoffs) == 1
    assert handoffs[0]["id"] in text_of(result) or handoffs[0]["id"] in json.dumps(result.structured_content or {})


def test_an_operator_server_can_write_inside_shop(srv, scratch, audit_log):
    result = call(srv.build_server("operator"), "write_file", {"path": scratch, "content": "hello"})
    assert not result.is_error, text_of(result)
    assert (REPO / scratch).read_text() == "hello"
    assert not events(audit_log, "permission_denied")


def test_an_ordinary_tool_error_reaches_the_client(srv):
    result = call(srv.build_server(), "read_file", {"path": "shop/no_such_file_step5.py"})
    assert result.is_error
    message = text_of(result)
    assert "not exist" in message.lower() or "not found" in message.lower(), (
        f"the model only sees {message!r}: an uncaught exception's text stays on the server. "
        "Look at how the SDK treats the exceptions a tool raises."
    )


# ==========================================================================
# 4. The server enforces permissions on its own
# ==========================================================================
def test_an_investigator_server_denies_write_file(srv, scratch, audit_log):
    result = call(srv.build_server("investigator"), "write_file", {"path": scratch, "content": SENTINEL})
    assert result.is_error
    assert not (REPO / scratch).exists(), "the denied write ran"
    message = text_of(result)
    assert "permission_denied" in message
    assert "create_handoff" in message, "an investigator's denied write should point at create_handoff"


def test_the_default_server_is_an_investigator(srv, scratch):
    result = call(srv.build_server(), "write_file", {"path": scratch, "content": SENTINEL})
    assert result.is_error and not (REPO / scratch).exists()


def test_a_server_denial_writes_exactly_one_audit_line_with_the_arguments(srv, scratch, audit_log):
    call(srv.build_server(), "write_file", {"path": scratch, "content": SENTINEL})
    denials = events(audit_log, "permission_denied")
    assert len(denials) == 1
    d = denials[0]
    assert d["tool"] == "write_file" and d["level"] == "investigator"
    assert SENTINEL in json.dumps(d.get("arguments")), "the audit line should keep the call's arguments"


def test_the_model_facing_denial_does_not_echo_the_arguments(srv, scratch):
    result = call(srv.build_server(), "write_file", {"path": scratch, "content": SENTINEL})
    assert SENTINEL not in text_of(result)


@pytest.mark.parametrize("level", ["investigator", "operator"])
@pytest.mark.parametrize("name, arguments", [
    ("read_file", {"path": "shop/../.env"}),
    ("read_file", {"path": "/etc/passwd"}),
    ("write_file", {"path": "../data/shop.db", "content": SENTINEL}),
    ("write_file", {"path": "shop/../pyproject.toml", "content": SENTINEL}),
])
def test_out_of_scope_paths_are_refused_with_no_handoff(srv, audit_log, level, name, arguments):
    before = (REPO / "pyproject.toml").read_text()
    result = call(srv.build_server(level), name, arguments)
    assert result.is_error
    message = text_of(result)
    print(message)
    assert "permission_denied" in message
    assert "create_handoff" not in message, "out of scope at every level: refuse, don't suggest escalating"
    assert SENTINEL not in message
    assert (REPO / "pyproject.toml").read_text() == before
    assert len(events(audit_log, "permission_denied")) == 1


def test_allowed_calls_write_no_denial_lines(srv, shop_log, audit_log):
    server = srv.build_server()
    call(server, "search_logs", {"query": "mug"})
    call(server, "read_file", {"path": "shop/__init__.py"})
    assert not events(audit_log, "permission_denied")


# ==========================================================================
# 5. A real transport: stdio, the way the Inspector launches it
# ==========================================================================
def stdio_session(level, audit_log, fn):
    from mcp import Client, StdioServerParameters

    env = {**os.environ, "AGENT_LOG": str(audit_log)}
    if level is not None:
        env["AGENT_LEVEL"] = level
    else:
        env.pop("AGENT_LEVEL", None)
    params = StdioServerParameters(command=sys.executable, args=["-m", "agent.mcp_server"], env=env, cwd=str(REPO))

    async def go():
        with anyio.fail_after(60):
            async with Client(params) as client:
                return await fn(client)

    return anyio.run(go)


def test_module_serves_over_stdio(audit_log):
    result = stdio_session(None, audit_log, lambda c: c.list_tools())
    assert {"search_logs", "read_file", "write_file", "create_handoff"} <= {t.name for t in result.tools}


def test_stdio_server_defaults_to_investigator(audit_log, scratch):
    result = stdio_session(None, audit_log, lambda c: c.call_tool("write_file", {"path": scratch, "content": SENTINEL}))
    assert result.is_error and "permission_denied" in text_of(result)
    assert not (REPO / scratch).exists()
    assert events(audit_log, "permission_denied"), "the server process should log to AGENT_LOG"


def test_stdio_server_reads_its_level_from_the_environment(audit_log, scratch):
    result = stdio_session("operator", audit_log, lambda c: c.call_tool("write_file", {"path": scratch, "content": "ok"}))
    assert not result.is_error, text_of(result)
    assert (REPO / scratch).read_text() == "ok"


def test_stdio_server_refuses_to_start_at_an_unknown_level(audit_log):
    env = {**os.environ, "AGENT_LOG": str(audit_log), "AGENT_LEVEL": "admin"}
    proc = subprocess.run([sys.executable, "-m", "agent.mcp_server"], cwd=REPO, env=env,
                          input="", capture_output=True, text=True, timeout=30)
    assert proc.returncode != 0, "a server at an unknown level should fail at startup, not serve"


# ==========================================================================
# 6. The client side: converting between MCP and Claude formats
# ==========================================================================
def test_tool_definitions_come_from_the_server(srv, cli):
    listed = list_tools(srv.build_server())
    defs = cli.tool_definitions_from_mcp(listed)
    assert [d["name"] for d in defs] == [t.name for t in listed.tools]
    for d, t in zip(defs, listed.tools):
        assert set(d) == {"name", "description", "input_schema"}
        assert d["description"] == t.description
        assert d["input_schema"] == t.input_schema
    json.dumps(defs)


def test_to_tool_result_success(srv, cli, shop_log):
    result = call(srv.build_server(), "search_logs", {"query": "mug"})
    block = cli.to_tool_result("toolu_1", result)
    assert block["type"] == "tool_result" and block["tool_use_id"] == "toolu_1"
    assert block["is_error"] is False
    assert isinstance(block["content"], str) and "mug" in block["content"]
    json.dumps(block)


def test_to_tool_result_carries_errors(srv, cli, scratch):
    result = call(srv.build_server(), "write_file", {"path": scratch, "content": SENTINEL})
    block = cli.to_tool_result("toolu_2", result)
    assert block["is_error"] is True
    assert "permission_denied" in block["content"]


# ==========================================================================
# 7. The loop as an MCP client
# ==========================================================================
def test_run_with_mcp_hands_the_model_the_servers_tools(srv, cli):
    server = srv.build_server()
    model = ScriptedModel(answers("nothing to do"))
    out = cli.run_with_mcp("hi", model, server)
    assert out["stopped"] == "answer" and out["answer"] == "nothing to do" and out["turns"] == 1
    names = {d["name"] for d in model.tools_seen[0]}
    assert {"search_logs", "read_file", "write_file", "create_handoff"} <= names
    assert model.calls[0] == [{"role": "user", "content": "hi"}]


def test_run_with_mcp_reruns_the_step4_scenario(srv, cli, scratch, audit_log):
    """Denied write, then a handoff, then an answer: Step 4's scenario, through MCP."""
    model = ScriptedModel(
        asks(tool_use("t1", "write_file", {"path": scratch, "content": SENTINEL})),
        asks(tool_use("t2", "create_handoff", {
            "summary": "Needs a write", "evidence": ["denied write_file"], "requested_action": f"write {scratch}"})),
        answers("Filed a handoff."),
    )
    out = cli.run_with_mcp("make the change", model, srv.build_server("investigator"))
    assert out["stopped"] == "answer" and out["turns"] == 3
    assert not (REPO / scratch).exists()

    denial = results_in(out["messages"][2])
    assert [r["tool_use_id"] for r in denial] == ["t1"]
    assert denial[0]["is_error"] is True and "permission_denied" in denial[0]["content"]
    assert SENTINEL not in denial[0]["content"]

    handoff = results_in(out["messages"][4])
    assert [r["tool_use_id"] for r in handoff] == ["t2"] and not handoff[0].get("is_error")

    assert len(events(audit_log, "permission_denied")) == 1, "a denial is logged once, by the server"
    assert len(events(audit_log, "handoff")) == 1


def test_run_with_mcp_adds_no_gate_of_its_own(srv, cli, scratch, audit_log):
    model = ScriptedModel(
        asks(tool_use("t1", "write_file", {"path": scratch, "content": "from operator"})),
        answers("done"),
    )
    out = cli.run_with_mcp("write it", model, srv.build_server("operator"))
    assert out["stopped"] == "answer"
    assert (REPO / scratch).read_text() == "from operator", (
        "an operator server allowed this write; the client shouldn't second-guess it with a level of its own"
    )
    assert not events(audit_log, "permission_denied")


def test_run_with_mcp_sends_every_result_of_a_turn_in_one_message(srv, cli, shop_log):
    model = ScriptedModel(
        asks(
            tool_use("a", "search_logs", {"query": "mug"}),
            tool_use("b", "read_file", {"path": "shop/../.env"}),
            tool_use("c", "read_file", {"path": "shop/__init__.py"}),
        ),
        answers("ok"),
    )
    out = cli.run_with_mcp("look", model, srv.build_server())
    results = results_in(out["messages"][2])
    assert [r["tool_use_id"] for r in results] == ["a", "b", "c"]
    assert [bool(r.get("is_error")) for r in results] == [False, True, False]
    assert len(out["messages"]) == 4


def test_run_with_mcp_survives_an_unknown_tool(srv, cli):
    model = ScriptedModel(asks(tool_use("x", "delete_everything", {})), answers("ok"))
    out = cli.run_with_mcp("go", model, srv.build_server())
    r = results_in(out["messages"][2])[0]
    assert r["tool_use_id"] == "x" and r["is_error"] is True
    assert out["stopped"] == "answer"


def test_run_with_mcp_stops_at_max_turns(srv, cli, shop_log):
    model = ScriptedModel(*[asks(tool_use(f"t{i}", "search_logs", {"query": "mug"})) for i in range(3)])
    out = cli.run_with_mcp("loop", model, srv.build_server(), max_turns=3)
    assert out["stopped"] == "max_turns" and out["turns"] == 3
    assert isinstance(out["answer"], str) and out["answer"].strip()


def test_run_with_mcp_answer_on_the_last_turn_is_an_answer(srv, cli, shop_log):
    model = ScriptedModel(asks(tool_use("t1", "search_logs", {"query": "mug"})), answers("found it"))
    out = cli.run_with_mcp("go", model, srv.build_server(), max_turns=2)
    assert out["stopped"] == "answer" and out["answer"] == "found it"


# ==========================================================================
# 8. Carry-forwards from the Step 4 review
# ==========================================================================
@pytest.mark.parametrize("level", ["investigator", "operator"])
def test_scope_is_checked_before_level(perms, level):
    denial = perms.check(perms.Context(level), "write_file", {"path": "../data/shop.db", "content": ""})
    assert denial is not None
    assert "create_handoff" not in denial["suggestion"], "out of scope: refuse, don't suggest a handoff"


def test_an_investigator_write_inside_shop_still_suggests_a_handoff(perms):
    denial = perms.check(perms.Context(), "write_file", {"path": "shop/app.py", "content": ""})
    assert "create_handoff" in denial["suggestion"]


def test_model_facing_denial_has_no_arguments(perms):
    denial = perms.check(perms.Context(), "write_file", {"path": "shop/app.py", "content": SENTINEL})
    assert "arguments" not in denial and SENTINEL not in json.dumps(denial)


@pytest.mark.parametrize("path", [None, 42, ["shop/app.py"], {"p": 1}])
def test_a_non_string_path_is_denied_not_a_crash(perms, path):
    denial = perms.check(perms.Context("operator"), "read_file", {"path": path})
    assert denial is not None and denial["error"] == "permission_denied"


def test_loop_answer_on_the_last_turn_is_an_answer(loop, shop_log):
    responses = [asks(tool_use("t1", "search_logs", {"query": "mug"})), answers("done")]

    def model(messages):
        return responses.pop(0)

    out = loop.run("go", model, max_turns=2)
    assert out["stopped"] == "answer" and out["answer"] == "done"

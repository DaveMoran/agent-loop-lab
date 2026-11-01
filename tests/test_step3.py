"""Step 3 acceptance tests, part 1: the loop and the two new tools. No API calls.

Part 2 (evidence from one real run) is tests/test_step3_artifacts.py.
Don't edit these files; if a test looks wrong, raise it and we'll change it together.

What these tests pin (the contract), and nothing more:

agent/tools.py  (adds to Step 2)
    read_file(path: str) -> str
        - Returns the file's text. Only files inside the shop/ package are allowed.
        - A relative path is resolved against the repo root (the folder that contains
          shop/), so "shop/app.py" works from any working directory.
        - Anything that resolves outside shop/ raises PermissionError: ".env",
          "shop/../.env", "agent/tools.py", absolute paths elsewhere, and look-alike
          siblings such as "<repo>/shop_evil/x.py". A missing file inside shop/ raises
          FileNotFoundError.
    open_readonly_db() -> sqlite3.Connection
        - Opens the database at shop.db.db_path() (honours SHOP_DB, read at call time)
          READ-ONLY at the connection level, so a write fails even if it got past the
          SELECT check. Raises FileNotFoundError if the database doesn't exist.
    run_query(sql: str) -> list[dict]
        - Runs one SELECT through open_readonly_db() and returns the rows as dicts.
        - Anything that isn't a SELECT raises PermissionError, before touching the db.
    READ_FILE_PARAMETERS, RUN_QUERY_PARAMETERS: dict    (JSON Schema, like Step 2's)
    TOOLS: dict[str, Callable]
        - The default registry: {"search_logs": ..., "read_file": ..., "run_query": ...}

agent/claude_raw.py  (adds to Step 2)
    tool_definitions() -> list[dict]
        - Claude tool definitions for every tool in TOOLS, using the shared schemas.

agent/loop.py
    run(prompt: str, call_model, tools: dict | None = None, max_turns: int = 10) -> dict
        - call_model(messages: list) -> dict is injected: it takes the history and
          returns one raw Claude Messages response as a dict. The loop never imports a
          provider SDK; your real call_model (in a script) wraps the API.
        - tools defaults to agent.tools.TOOLS.
        - The loop owns the history. It starts as [{"role": "user", "content": prompt}],
          appends each assistant turn exactly as returned (content unchanged), and
          appends ONE user message per turn holding a tool_result for every tool_use,
          in the same order.
        - A tool that raises, an unknown tool name, or bad arguments does not crash the
          run: it becomes a tool_result with "is_error": True and the error text.
        - Ends when the model answers without a tool_use, or when it has called the
          model max_turns times. It never raises for either.
        - Returns {"answer": str, "messages": list, "turns": int, "stopped": str}:
            stopped == "answer"     answer is the final text
            stopped == "max_turns"  answer is a clear message that names the limit
          turns is the number of model calls made.

How you structure the code beyond these names is up to you. Keep loop.py small
(see the size test): the plan says about 60 lines, and the build post quotes it.
"""
import copy
import importlib
import json
import sqlite3
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SHOP_DIR = REPO / "shop"


@pytest.fixture
def tools():
    return importlib.import_module("agent.tools")


@pytest.fixture
def claude():
    return importlib.import_module("agent.claude_raw")


@pytest.fixture
def loop():
    return importlib.import_module("agent.loop")


# --------------------------------------------------------------------------
# A scripted stand-in for the model: returns canned Claude responses in order
# and records a snapshot of the history it was given on each call.
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


# ==========================================================================
# 1. read_file: limited to shop/
# ==========================================================================
def test_read_file_reads_a_shop_file(tools):
    assert "def checkout" in tools.read_file("shop/app.py")


def test_read_file_works_from_any_working_directory(tools, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert "def checkout" in tools.read_file("shop/app.py")


def test_read_file_accepts_an_absolute_path_inside_shop(tools):
    assert "def checkout" in tools.read_file(str(SHOP_DIR / "app.py"))


@pytest.mark.parametrize("path", [
    ".env",
    "agent/tools.py",
    "shop/../.env",
    "shop/../agent/tools.py",
    "/etc/passwd",
    str(REPO / "pyproject.toml"),
    str(SHOP_DIR) + "_evil/x.py",   # a sibling whose name merely starts with "shop"
])
def test_read_file_refuses_anything_outside_shop(tools, path):
    with pytest.raises(PermissionError):
        tools.read_file(path)


def test_read_file_missing_file_inside_shop_raises(tools):
    with pytest.raises(FileNotFoundError):
        tools.read_file("shop/does_not_exist.py")


# ==========================================================================
# 2. run_query: SELECT only, on a read-only connection
# ==========================================================================
@pytest.fixture
def shop_db(tmp_path, monkeypatch):
    from shop import db

    path = tmp_path / "shop.db"
    conn = db.connect(str(path))
    db.seed(conn)
    conn.execute("INSERT INTO orders (user_id, product_id, quantity, total, status) VALUES (3, 1, 2, 50.0, 'paid')")
    conn.commit()
    conn.close()
    monkeypatch.setenv("SHOP_DB", str(path))
    return path


def count(path, table):
    c = sqlite3.connect(path)
    try:
        return c.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    finally:
        c.close()


def test_run_query_returns_rows_as_dicts(tools, shop_db):
    rows = tools.run_query("SELECT id, name FROM products ORDER BY id LIMIT 2")
    assert rows == [{"id": 1, "name": "Mug"}, {"id": 2, "name": "Poster"}]


def test_run_query_is_case_and_whitespace_tolerant(tools, shop_db):
    assert tools.run_query("  select count(*) as n from orders")[0]["n"] == 1


@pytest.mark.parametrize("sql", [
    "DELETE FROM orders",
    "UPDATE products SET stock = 0",
    "INSERT INTO users VALUES (99, 'x', 'x@x')",
    "DROP TABLE orders",
    "PRAGMA writable_schema = 1",
    "ATTACH DATABASE 'other.db' AS other",
])
def test_run_query_refuses_anything_but_select(tools, shop_db, sql):
    with pytest.raises(PermissionError):
        tools.run_query(sql)
    assert count(shop_db, "orders") == 1 and count(shop_db, "products") == 15


def test_run_query_cannot_smuggle_a_write_after_a_select(tools, shop_db):
    assert callable(getattr(tools, "run_query", None)), "run_query isn't defined yet"
    with pytest.raises(Exception):
        tools.run_query("SELECT 1; DELETE FROM orders")
    assert count(shop_db, "orders") == 1


def test_readonly_connection_refuses_writes_on_its_own(tools, shop_db):
    """Defence in depth: even with the SELECT check bypassed, the connection can't write."""
    conn = tools.open_readonly_db()
    try:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("DELETE FROM orders")
    finally:
        conn.close()
    assert count(shop_db, "orders") == 1


def test_open_readonly_db_missing_database_raises(tools, tmp_path, monkeypatch):
    monkeypatch.setenv("SHOP_DB", str(tmp_path / "nope.db"))
    with pytest.raises(FileNotFoundError):
        tools.open_readonly_db()
    assert not (tmp_path / "nope.db").exists(), "opening must not create an empty database"


# ==========================================================================
# 3. Schemas, registry and Claude definitions
# ==========================================================================
@pytest.mark.parametrize("name,required", [
    ("READ_FILE_PARAMETERS", ["path"]),
    ("RUN_QUERY_PARAMETERS", ["sql"]),
])
def test_new_schemas_describe_their_arguments(tools, name, required):
    schema = getattr(tools, name)
    assert schema["type"] == "object"
    assert schema.get("required") == required
    for prop_name, prop in schema["properties"].items():
        assert prop.get("description"), f"{prop_name} needs a description: the model reads it"


def test_registry_holds_the_three_tools(tools):
    # ">=" since Step 4, which adds write_file and create_handoff to the registry.
    assert set(tools.TOOLS) >= {"search_logs", "read_file", "run_query"}
    assert tools.TOOLS["read_file"] is tools.read_file
    assert tools.TOOLS["run_query"] is tools.run_query
    assert tools.TOOLS["search_logs"] is tools.search_logs


def test_claude_definitions_cover_the_registry(claude, tools):
    defs = {d["name"]: d for d in claude.tool_definitions()}
    assert set(defs) == set(tools.TOOLS)
    assert defs["search_logs"]["input_schema"] == tools.SEARCH_LOGS_PARAMETERS
    assert defs["read_file"]["input_schema"] == tools.READ_FILE_PARAMETERS
    assert defs["run_query"]["input_schema"] == tools.RUN_QUERY_PARAMETERS
    assert all(d.get("description") for d in defs.values())


# ==========================================================================
# 4. The loop
# ==========================================================================
def test_model_answers_straight_away(loop):
    model = ScriptedModel(answers("All good."))
    out = loop.run("hi", model, tools={})
    assert out["stopped"] == "answer"
    assert out["answer"] == "All good."
    assert out["turns"] == 1
    assert model.calls[0] == [{"role": "user", "content": "hi"}]


def test_one_tool_call_then_answer(loop):
    seen = []

    def echo(text):
        seen.append(text)
        return {"echo": text}

    model = ScriptedModel(asks(tool_use("t1", "echo", {"text": "ping"})), answers("Done."))
    out = loop.run("go", model, tools={"echo": echo})
    assert seen == ["ping"], "your code, not the model, runs the tool"
    assert out == {**out, "stopped": "answer", "answer": "Done.", "turns": 2}
    [result] = results_in(model.calls[1][-1])
    assert result["tool_use_id"] == "t1"
    assert json.loads(result["content"]) == {"echo": "ping"}
    assert not result.get("is_error")


def test_history_replays_the_assistant_turn_unchanged(loop):
    thinking = {"type": "thinking", "thinking": "", "signature": "sig=="}
    first = asks(thinking, tool_use("t1", "echo", {"text": "a"}))
    model = ScriptedModel(first, answers("ok"))
    loop.run("go", model, tools={"echo": lambda text: text})
    history = model.calls[1]
    assert [m["role"] for m in history] == ["user", "assistant", "user"]
    assert history[1]["content"] == first["content"], "append the assistant content exactly as returned"


def test_parallel_calls_go_back_in_one_message_in_order(loop):
    model = ScriptedModel(
        asks(tool_use("a", "echo", {"text": "1"}), tool_use("b", "echo", {"text": "2"})),
        answers("ok"),
    )
    loop.run("go", model, tools={"echo": lambda text: text})
    history = model.calls[1]
    assert len(history) == 3, "all results for one turn belong in a single user message"
    assert [r["tool_use_id"] for r in results_in(history[-1])] == ["a", "b"]


def test_returned_messages_are_the_full_history(loop):
    model = ScriptedModel(asks(tool_use("t1", "echo", {"text": "a"})), answers("ok"))
    out = loop.run("go", model, tools={"echo": lambda text: text})
    assert [m["role"] for m in out["messages"]] == ["user", "assistant", "user", "assistant"]
    assert out["messages"][-1]["content"] == answers("ok")["content"]


def test_a_throwing_tool_becomes_an_error_result_not_a_crash(loop):
    def broken(**_):
        raise RuntimeError("disk on fire")

    model = ScriptedModel(asks(tool_use("t1", "broken", {})), answers("The tool failed."))
    out = loop.run("go", model, tools={"broken": broken})
    assert out["stopped"] == "answer"
    [result] = results_in(model.calls[1][-1])
    assert result["is_error"] is True
    assert "disk on fire" in json.dumps(result["content"])


def test_an_unknown_tool_becomes_an_error_result(loop):
    model = ScriptedModel(asks(tool_use("t1", "rm_rf", {})), answers("ok"))
    out = loop.run("go", model, tools={})
    assert out["stopped"] == "answer"
    [result] = results_in(model.calls[1][-1])
    assert result["is_error"] is True and "rm_rf" in json.dumps(result["content"])


def test_bad_arguments_become_an_error_result(loop):
    model = ScriptedModel(asks(tool_use("t1", "echo", {"wrong": 1})), answers("ok"))
    loop.run("go", model, tools={"echo": lambda text: text})
    [result] = results_in(model.calls[1][-1])
    assert result["is_error"] is True


def test_max_turn_guard_returns_instead_of_raising(loop):
    model = ScriptedModel(*[asks(tool_use(f"t{i}", "echo", {"text": "again"})) for i in range(3)])
    out = loop.run("go", model, tools={"echo": lambda text: text}, max_turns=3)
    assert len(model.calls) == 3, "stop calling the model once the limit is reached"
    assert out["stopped"] == "max_turns"
    assert out["turns"] == 3
    assert out["answer"].strip() and "3" in out["answer"], "say clearly that the turn limit was hit"


def test_each_run_owns_its_own_history(loop):
    first = ScriptedModel(asks(tool_use("t1", "echo", {"text": "a"})), answers("one"))
    loop.run("first", first, tools={"echo": lambda text: text})
    second = ScriptedModel(answers("two"))
    loop.run("second", second, tools={})
    assert second.calls[0] == [{"role": "user", "content": "second"}], "history leaked between runs"


def test_default_tools_are_the_registry(loop, tools, shop_db):
    model = ScriptedModel(asks(tool_use("t1", "run_query", {"sql": "SELECT COUNT(*) AS n FROM orders"})), answers("1"))
    loop.run("how many orders?", model)
    [result] = results_in(model.calls[1][-1])
    assert not result.get("is_error")
    assert json.loads(result["content"]) == [{"n": 1}]


def test_loop_does_not_import_a_provider_sdk():
    source = (REPO / "agent" / "loop.py").read_text()
    for sdk in ("anthropic", "openai", "httpx", "requests"):
        assert f"import {sdk}" not in source and f"from {sdk}" not in source, (
            f"loop.py imports {sdk}: inject the model call instead, so the loop stays provider-free and testable"
        )


def test_loop_stays_small():
    lines = [
        line for line in (REPO / "agent" / "loop.py").read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert len(lines) <= 100, f"loop.py has {len(lines)} non-blank, non-comment lines; aim for about 60"

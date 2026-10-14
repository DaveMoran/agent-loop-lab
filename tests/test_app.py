import json

import pytest

from shop import app, db


@pytest.fixture()
def conn(tmp_path, monkeypatch):
    monkeypatch.setenv("SHOP_LOG", str(tmp_path / "app.log"))
    monkeypatch.delenv("SHOP_FAULTS", raising=False)
    c = db.connect()
    db.seed(c)
    return c


def stock(conn, pid):
    return conn.execute("SELECT stock FROM products WHERE id=?", (pid,)).fetchone()["stock"]


def test_seed_is_deterministic():
    a, b = db.connect(), db.connect()
    db.seed(a); db.seed(b)
    assert [tuple(r) for r in a.execute("SELECT * FROM products")] == [tuple(r) for r in b.execute("SELECT * FROM products")]


def test_checkout_decrements_stock_once(conn):
    before = stock(conn, 1)
    assert app.checkout(conn, 1, 1, 1)["ok"]
    assert stock(conn, 1) == before - 1


def test_double_decrement_fault(conn, monkeypatch):
    monkeypatch.setenv("SHOP_FAULTS", "double_decrement")
    before = stock(conn, 1)
    app.checkout(conn, 1, 1, 1)
    assert stock(conn, 1) == before - 2


def test_checkout_timeout_fault_logs_error(conn, monkeypatch, tmp_path):
    monkeypatch.setenv("SHOP_FAULTS", "checkout_timeout")
    conn.execute("UPDATE products SET price=150 WHERE id=1")
    assert app.checkout(conn, 1, 1, 1)["reason"] == "payment_timeout"
    records = [json.loads(l) for l in (tmp_path / "app.log").read_text().splitlines()]
    assert records[-1]["event"] == "payment_timeout"


def test_slow_search_fault_warns(conn, monkeypatch, tmp_path):
    monkeypatch.setenv("SHOP_FAULTS", "slow_search")
    app.search_products(conn, "Mug")
    records = [json.loads(l) for l in (tmp_path / "app.log").read_text().splitlines()]
    assert records[-1]["level"] == "WARN"

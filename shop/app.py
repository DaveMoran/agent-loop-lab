"""The shop. Small on purpose: the agent reads this code and its logs, it does not run the business."""
import sqlite3
import time

from . import faults
from .logs import emit


def search_products(conn: sqlite3.Connection, term: str) -> list[dict]:
    start = time.time()
    if "slow_search" in faults.active():
        time.sleep(0.05)  # stands in for a missing index; reported as a long duration
        duration_ms = 2400
    else:
        duration_ms = int((time.time() - start) * 1000) + 3
    rows = conn.execute("SELECT * FROM products WHERE name LIKE ?", (f"%{term}%",)).fetchall()
    level = "WARN" if duration_ms > 1000 else "INFO"
    emit(level, "search", term=term, results=len(rows), duration_ms=duration_ms)
    return [dict(r) for r in rows]


def checkout(conn: sqlite3.Connection, user_id: int, product_id: int, quantity: int) -> dict:
    product = conn.execute("SELECT * FROM products WHERE id=?", (product_id,)).fetchone()
    if product is None:
        emit("ERROR", "checkout_failed", user_id=user_id, product_id=product_id, reason="no_such_product")
        raise ValueError("no such product")
    if product["stock"] < quantity:
        emit("WARN", "checkout_rejected", user_id=user_id, product_id=product_id, reason="out_of_stock")
        return {"ok": False, "reason": "out_of_stock"}
    total = round(product["price"] * quantity, 2)
    if "checkout_timeout" in faults.active() and total > 100:
        emit("ERROR", "payment_timeout", user_id=user_id, product_id=product_id,
             total=total, upstream="payments", timeout_s=30)
        return {"ok": False, "reason": "payment_timeout"}
    decrement = quantity * (2 if "double_decrement" in faults.active() else 1)
    conn.execute("UPDATE products SET stock = stock - ? WHERE id=?", (decrement, product_id))
    cur = conn.execute("INSERT INTO orders (user_id, product_id, quantity, total, status) VALUES (?,?,?,?,?)",
                       (user_id, product_id, quantity, total, "paid"))
    conn.commit()
    emit("INFO", "order_created", order_id=cur.lastrowid, user_id=user_id, product_id=product_id, total=total)
    return {"ok": True, "order_id": cur.lastrowid, "total": total}


def get_order(conn: sqlite3.Connection, order_id: int) -> dict | None:
    row = conn.execute("SELECT * FROM orders WHERE id=?", (order_id,)).fetchone()
    return dict(row) if row else None

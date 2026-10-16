"""Tiny deterministic SQLite store. Same seed, same data, every time."""
import os
import random
import sqlite3
from pathlib import Path

NAMES = ["Mug", "Poster", "Tote Bag", "Sticker Pack", "Notebook", "Lamp", "Hoodie",
         "Water Bottle", "Cap", "Candle", "Plant Pot", "Keychain", "Desk Mat", "Socks", "Pen Set"]


def db_path() -> Path:
    """Where the simulator persists the store: SHOP_DB, or data/shop.db. Read at call time."""
    return Path(os.environ.get("SHOP_DB", "data/shop.db"))


def connect(path: str = ":memory:") -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def seed(conn: sqlite3.Connection, seed_value: int = 7) -> None:
    rng = random.Random(seed_value)
    conn.executescript("""
        DROP TABLE IF EXISTS products; DROP TABLE IF EXISTS users; DROP TABLE IF EXISTS orders;
        CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT, price REAL, stock INTEGER);
        CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, email TEXT);
        CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER, product_id INTEGER,
                             quantity INTEGER, total REAL, status TEXT);
    """)
    for i, name in enumerate(NAMES, start=1):
        conn.execute("INSERT INTO products VALUES (?,?,?,?)",
                     (i, name, round(rng.uniform(4, 120), 2), rng.randint(5, 60)))
    for i in range(1, 21):
        conn.execute("INSERT INTO users VALUES (?,?,?)", (i, f"User {i}", f"user{i}@example.com"))
    conn.commit()

"""Generate traffic so there is something in the logs. Usage:
    SHOP_FAULTS=checkout_timeout python -m shop.simulate --n 200

The store is reseeded on every run and persisted to SHOP_DB (default data/shop.db),
so the agent can query it afterwards.
"""
import argparse
import random

from . import app, db


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()
    rng = random.Random(args.seed)
    path = db.db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = db.connect(str(path))
    db.seed(conn)
    for _ in range(args.n):
        if rng.random() < 0.4:
            app.search_products(conn, rng.choice(db.NAMES).split()[0])
        else:
            app.checkout(conn, rng.randint(1, 20), rng.randint(1, len(db.NAMES)), rng.randint(1, 3))
    conn.commit()
    conn.close()
    print(f"simulated {args.n} requests (db: {path})")


if __name__ == "__main__":
    main()

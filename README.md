# agent-loop-lab

A from-scratch agent loop in plain Python, plus a tiny shop app to point it at.

The shop is the **target**, not the agent's job. The agent is a developer-facing loop that reads the shop's
code and logs, investigates problems, and escalates to a human when it needs to write. Read-only by default.

## The shop (`shop/`)
Standard library only. Deterministic dummy data (15 products, 20 users). Every event is a JSON line in `logs/app.log`.

| Fault | Switch | What you see in the logs |
|---|---|---|
| `slow_search` | `SHOP_FAULTS=slow_search` | `search` events at WARN with `duration_ms` over 1000 |
| `checkout_timeout` | `SHOP_FAULTS=checkout_timeout` | `payment_timeout` errors, only on orders over $100 |
| `double_decrement` | `SHOP_FAULTS=double_decrement` | No error at all. Stock drops by twice the quantity sold |

The third one is deliberate: it leaves no error in the logs, so it can only be found by comparing orders to stock.
Combine faults with commas.

## Try it
```bash
SHOP_FAULTS=checkout_timeout python -m shop.simulate --n 200
tail -5 logs/app.log
python -m pytest
```

## Scenarios you will write (Step 7)
Developer tasks pointed at this app, for example: "support says some checkouts are failing, triage it"
(read-only investigation), "PM wants stock shown on search results" (a write, so it escalates).
# agent-loop-lab

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

## Setup
Needs Python 3.11+. The shop itself has no dependencies; the extras are for the agent and for tests.
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[agent,dev]"
cp .env.example .env    # then fill in OPENAI_API_KEY and ANTHROPIC_API_KEY
```
`.env` and `.venv/` are gitignored. The scripts in `scripts/` load `.env` themselves with `python-dotenv`; the tests
never need the keys.

## Try it
Run everything from the repo root with the venv active.

**Generate some logs.** Pick the faults you want to investigate (combine them with commas):
```bash
SHOP_FAULTS=slow_search,checkout_timeout python -m shop.simulate --n 200
tail -5 logs/app.log
```

**Run the tests.** They are offline and free: no API calls, no keys needed.
```bash
python -m pytest
```
The same suite runs in CI on every pull request and on pushes to `main` (`.github/workflows/tests.yml`).

**Run a real round trip.** Each runner makes live API calls (it costs tokens) and overwrites its files in
`artifacts/step2/`:
```bash
python scripts/openai_runner.py
python scripts/claude_runner.py
```
Run them after generating logs, so the tool has WARN entries to find. The saved payloads are the evidence the
Step 2 artifact tests check, so re-run both runners together and commit the results.

## Scenarios you will write (Step 7)
Developer tasks pointed at this app, for example: "support says some checkouts are failing, triage it"
(read-only investigation), "PM wants stock shown on search results" (a write, so it escalates).
# agent-loop-lab

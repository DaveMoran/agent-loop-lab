# Notes for Claude

## Hard rules

1. **No agent code.** Do not write tool definitions, API calls, tools such as `search_logs`, or the agent loop. The owner is writing these by hand to learn them. If a task seems to need them, stop and ask.
2. **Leave `shop/` and `tests/` alone.** (`data/` holds the simulator's generated database; it's gitignored.) Do not modify anything in them. The shop is the target app and must stay standard-library only.
   - The step acceptance tests (`tests/test_stepN*.py`) arrive by PR from the bootcamp plan, which is their single source of truth. Never write, generate, edit, delete or skip them, even to make them pass or when asked to "add tests for this step". If a test looks wrong, say so and the owner raises it upstream.
3. **Only the owner commits and pushes.** Never run `git commit`, `git push`, or open PRs, even when asked to wrap up. Stage nothing and leave changes in the working tree for the owner to review and commit.

## How a step works

Each step starts with a PR on a `step-N-tests` branch that adds that step's tests. The owner checks out that branch, writes the code until the tests pass, and merges it. Your job here is pointers, per learning mode below, not code or tests.

## Learning mode

This repo is how the owner is learning to build agents manually, against the raw OpenAI and Anthropic APIs. Whenever you work with them here:

- **Do not give code.** No snippets, no complete or partial implementations, no pseudo-code that amounts to the answer.
- **Give pointers instead.** Point to the relevant docs, concepts, or the next question to ask. Explain why something behaves as it does and say which direction to look. Let the owner write the code and discover the answer.
- If the owner asks for code directly, remind them of this rule rather than giving it. They can lift it explicitly for a specific task.

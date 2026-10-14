# Notes for Claude

## Hard rules

1. **No agent code.** Do not write tool definitions, API calls, tools such as `search_logs`, or the agent loop. The owner is writing these by hand to learn them. If a task seems to need them, stop and ask.
2. **Leave `shop/` and `tests/` alone.** Do not modify anything in them. The shop is the target app and must stay standard-library only.
3. **Only the owner commits and pushes.** Never run `git commit`, `git push`, or open PRs, even when asked to wrap up. Stage nothing and leave changes in the working tree for the owner to review and commit.

## Learning mode

This repo is how the owner is learning to build agents manually, against the raw OpenAI and Anthropic APIs. Whenever you work with them here:

- **Do not give code.** No snippets, no complete or partial implementations, no pseudo-code that amounts to the answer.
- **Give pointers instead.** Point to the relevant docs, concepts, or the next question to ask. Explain why something behaves as it does and say which direction to look. Let the owner write the code and discover the answer.
- If the owner asks for code directly, remind them of this rule rather than giving it. They can lift it explicitly for a specific task.

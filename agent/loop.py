def run(prompt: str, call_model, tools: dict | None = None, max_turns: int = 10) -> dict:
    stopped = "answer"
    
    model_input = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    turns = 0

    model_response = call_model(model_input)
    model_input += model_response
    turns += 1

    return {
        "answer": model_response["content"][0]["text"],
        "messages": [],
        "turns": turns,
        "stopped": stopped
    }
from anthropic import Anthropic
from dotenv import load_dotenv
from agent.claude_raw import tool_definition

if __name__ == "__main__":

    load_dotenv()

    client = Anthropic()

    messages = [{"role": "user", "content": "Give me all WARN logs"}]

    response = client.messages.create(
        model="claude-opus-5-5",
        max_tokens=1024,
        tools=[tool_definition()],
        messages=messages
    )

    #open and read the file after the appending:
    with open("artifacts/step2/anthropic_tool_request.json", "w") as f:
        f.write(response.model_dump_json(indent=2))
        
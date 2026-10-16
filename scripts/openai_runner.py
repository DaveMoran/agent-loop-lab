import json
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from agent.openai_raw import extract_tool_calls, tool_definition, tool_result
from agent.tools import search_logs

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts" / "step2"
OPENAI_FILE = ARTIFACTS / "openai_tool_request.json"
OPENAI_SECOND = ARTIFACTS / "openai_second_request.json"
OPENAI_FINAL = ARTIFACTS / "openai_final_response.json"

if __name__ == "__main__":

    load_dotenv()
    client = OpenAI()

    input_list = [{"role": "user", "content": "Give me all logs that are warnings"}]

    response = client.responses.create(
        model="gpt-6-astra",
        tools=[tool_definition()],
        input=input_list,
    )

    with open(OPENAI_FILE, "w") as f:
        f.write(response.model_dump_json(indent=2))

    input_list += response.output
    tool_calls = extract_tool_calls(response.model_dump())

    for call in tool_calls:
        result = tool_result(call["id"], search_logs(**call["arguments"]))
        input_list.append(result)

    with open(OPENAI_SECOND, "w") as f:
        clean_input = [
            item.model_dump() if hasattr(item, 'model_dump') else item 
            for item in input_list
        ]

        f.write(json.dumps({ "input": clean_input }))

    second_response = client.responses.create(
        model="gpt-6-astra",
        tools=[tool_definition()],
        input=input_list,
    )

    with open(OPENAI_FINAL, "w") as f:
        f.write(second_response.model_dump_json(indent=2))
from openai import OpenAI
from dotenv import load_dotenv

from agent.openai_raw import tool_definition, extract_tool_calls, tool_result
from agent.tools import search_logs

import json

if __name__ == "__main__":

    load_dotenv()

    client = OpenAI()

    input_list = [{"role": "user", "content": "Give me all logs that are warnings"}]

    response = client.responses.create(
        model="gpt-6-astra",
        tools=[tool_definition()],
        input=input_list,
    )

    # open and read the file after the appending:
    with open("artifacts/step2/openai_tool_request.json", "w") as f:
        f.write(response.model_dump_json(indent=2))

    # Save function call outputs for subsequent requests
    input_list += response.output

    tool_calls = extract_tool_calls(response.model_dump())

    results = []

    for call in tool_calls:        
        result = tool_result(call["id"], search_logs(**call["arguments"]))
        input_list.append(result)

    # open and read the file after the appending:
    with open("artifacts/step2/openai_second_request.json", "w") as f:
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
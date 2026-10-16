from openai import OpenAI
from dotenv import load_dotenv
from agent.openai_raw import tool_definition

if __name__ == "__main__":

    load_dotenv()

    client = OpenAI()

    input_list = [{"role": "user", "content": "Give me all logs that are warnings"}]

    response = client.responses.create(
        model="gpt-6-astra",
        tools=[tool_definition()],
        input=input_list,
    )

    #open and read the file after the appending:
    with open("artifacts/step2/openai_tool_request.json", "w") as f:
        f.write(response.model_dump_json(indent=2))
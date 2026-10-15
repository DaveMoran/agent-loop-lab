"""
Individual tools that the agent will call that will allow it to go back and forth with the client while 
"""

from pprint import pprint
import json

from shop.logs import log_path

from difflib import SequenceMatcher, get_close_matches


def search_logs(query, level="ALL", limit=10):
    with open(log_path(), "r") as file:
        final_logs = []

        # Step 1: Filter out all queries
        for line in file:

            # If no query sent, send max results
            if not query:
                line_as_json = json.loads(line)
                final_logs.append(line_as_json)
            else:
                if query.casefold() in line.casefold():
                    line_as_json = json.loads(line)
                    final_logs.append(line_as_json)

        return final_logs

def main():
    # Call your function inside main
    pprint(search_logs("test", "ERROR", 3))

# The entry point condition
if __name__ == "__main__":
    main()
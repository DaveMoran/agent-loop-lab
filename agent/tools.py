"""
Individual tools that the agent will call that will allow it to go back and forth with the client while 
"""

from pprint import pprint
import json

from shop.logs import log_path

from difflib import SequenceMatcher, get_close_matches


def search_logs(query, level="ALL", limit=20):
    # Immediate return for 0 limit
    if limit < 1:
        return []
    
    with open(log_path(), "r") as file:
        final_logs = []
        # Step 1: Filter out all queries
        for line in file:
            is_match = False
            line_as_json = json.loads(line)

            # If no query sent, send max results
            if not query and level == "ALL":
                final_logs.append(line_as_json)
                continue
            else:
                # If query AND level sent
                if query and level != "ALL":
                    is_match = matches_query(query, line) and matches_level(level, line_as_json)
                else:
                    # if query
                    if query and matches_query(query, line):
                        is_match = True

                    # if level
                    if level != "ALL":
                        if matches_level(level, line_as_json):
                            is_match = True
                    

            if is_match:
                final_logs.append(line_as_json)

        return final_logs[-limit:]

def matches_query(query, line):
    return query.casefold() in line.casefold()

def matches_level(level, json):
    return level.casefold() == json["level"].casefold()
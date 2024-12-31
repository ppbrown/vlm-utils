#!/bin/env python

# give list of json files on stdin.
# print out url item from each of those

import sys
import json

def process_json_files():
    for filename in sys.stdin:
        filename = filename.strip()

        try:
            with open(filename, 'r') as f:
                data = json.load(f)

            print(data.get("md5"))

        except (json.JSONDecodeError, FileNotFoundError):
            # Skip files that aren't valid JSON or can't be found
            continue

# Process each JSON file provided via stdin
process_json_files()

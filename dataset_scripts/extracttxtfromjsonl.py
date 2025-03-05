#!/bin/env python

"""
python program that is given the name of jsonl file and caption field as arguments.
Reads in all the json data from the jsonl file..
Then processes a list of .json files in stdin.
Rreads in "url" from the .json file
Finds matching object from original jsonl file
Extracts the field matching the caption field from object
Creates .txt filename based on the .json file, and writes the
caption field data to it.
"""


import json
import sys
import gzip
from pathlib import Path

# Return dictionary for entire jsonl file
def read_jsonl(file_path):
    """Read and parse a JSONL file."""
    if file_path.endswith(".gz"):
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            return [json.loads(line) for line in f]
    else:
        with open(file_path, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f]

def read_json(file_path):
    """Read and parse a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def find_matching_object(data_list, url):
    """Find the object in the JSONL data that matches the given URL."""
    for obj in data_list:
        if obj.get("url") == url:
            return obj
    return None

def main(jsonl_file,caption_field):
    # Read the JSONL data
    data_list = read_jsonl(jsonl_file)
    print("Read ",jsonl_file)

    # Process each .json file from stdin
    for line in sys.stdin:
        json_file_path = line.strip()
        if not json_file_path.endswith('.json'):
            print(f"Skipping non-JSON file: {json_file_path}", file=sys.stderr)
            continue

        # Read the JSON file and extract the URL
        try:
            json_data = read_json(json_file_path)
            url = json_data.get("url")
            if not url:
                print(f"No 'url' found in {json_file_path}", file=sys.stderr)
                continue

            # Find the matching object in the JSONL data
            matching_object = find_matching_object(data_list, url)
            if not matching_object:
                print(f"No matching object found for URL: {url}", file=sys.stderr)
                continue

            # Extract the 'caption_llava_short' field
            caption_llava_short = matching_object.get("caption_llava_short")
            if not caption_llava_short:
                print(f"No 'caption_llava_short' found for URL: {url}", file=sys.stderr)
                continue

            # Create the .txt file and write the data
            txt_file_path = Path(json_file_path).with_suffix('.txt')
            with open(txt_file_path, 'w', encoding='utf-8') as txt_file:
                txt_file.write(caption_llava_short)

            print(f"Processed: {json_file_path} -> {txt_file_path}")

        except Exception as e:
            print(f"Error processing {json_file_path}: {e}", file=sys.stderr)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <jsonl_file> <caption field name>", file=sys.stderr)
        sys.exit(1)

    jsonl_file = sys.argv[1]
    caption_field = sys.argv[2]
    main(jsonl_file,caption_field)



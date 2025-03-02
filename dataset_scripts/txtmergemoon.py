#!/bin/env python

# Change name as desired
FIELD="moondream" 


"""
Synopsis: allows adding of new captions to existing .jsonl file


This script is useful when you have downloaded a dataset and
have a corresponding .json file to each image file, which
includes the associated URL for each image.
Similarly, it assumes that you have an original .jsonl or .jsonl.gz file
(although you could modify it to simply output a brand new jsonl)

When you have generated NEW captions in .txt files for each image,
run this script to add or overwrite the captions associated with
each image.

Note: will only print lines actually matched from original jsonl file


## Usage:

    thisscript origdata.jsonl < list_of_imgfiles > new.jsonl


## Internals:
    Take name of jsonl file as arg.
    Read fully into memory
        (You should ideally be using a targetted subset jsonl file, not full laion2b)
    Take txt filenames from stdin.
    Read in text from txt file
    Open matching .json file, read in JSON object
    Merge text as FIELD entry
    Find match of JSON.url with JSONL url entry.
    Merge the two in memory
    Write out the merged JSONL line
"""



import json
import sys
import gzip
from pathlib import Path



# Return dictionary for entire jsonl file
def read_jsonl(file_path):
    """Read and parse a JSONL file."""
    try:
        if file_path.endswith(".gz"):
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                return [json.loads(line) for line in f]
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                return [json.loads(line.strip()) for line in f]

    except Exception as e:
        print(f"Error processing {file_path}: Format corrupt somewhere", file=sys.stderr)
        exit(0)

def find_matching_object(data_list, url):
    """Find the object in the JSONL data that matches the given URL."""
    for obj in data_list:
        if obj.get("url") == url:
            return obj
    return None


# Return a json.load object
def merge_json_and_txt(txt_file):
    try:
        with open(txt_file) as f:
            txt=f.read()
            txt=txt.strip()

        json_file = txt_file.rsplit('.', 1)[0] + '.json'
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            return

        data[FIELD] = txt

    except Exception as e:
        print(f"Error processing {txt_file}: {e}", file=sys.stderr)
    return data

def main(jsonl_file):
    # Read the JSONL data
    data_list = read_jsonl(jsonl_file)
    print("Read ",jsonl_file, file=sys.stderr)
    print("Object count: ",len(data_list), file=sys.stderr)

    # Process each .json file from stdin
    for line in sys.stdin:
        txt_file = line.strip()
        if not txt_file:
            continue
        json_data = merge_json_and_txt(txt_file)

        if not json_data:
            continue

        url = json_data.get("url")
        if not url:
            print(f"No matching url found for {txt_file}", file=sys.stderr)
            continue

        value = json_data.get(FIELD)
        if not value:
            print(f"Internal error: No '{FIELD}' found processing {txt_file}", file=sys.stderr)
            continue

        # Find the matching object in the JSONL data
        matching_object = find_matching_object(data_list, url)
        if not matching_object:
            print(f"No matching object found for URL: {url}", file=sys.stderr)
            continue
        matching_object[FIELD] = value

        print(json.dumps(matching_object))

        print(f"Processed: {txt_file}", file=sys.stderr)


    ################################# 
    print(f"Merging field {FIELD} from txt with json files", file=sys.stderr)
    for line in sys.stdin:
        txt_file = line.strip()
        if txt_file:
            merge_json_and_txt(txt_file)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <jsonl_file> < txt-file-list", file=sys.stderr)
        sys.exit(1)

    jsonl_file = sys.argv[1]
    main(jsonl_file)


#################################################


#!/bin/env python

"""
Usage:
    remove_tags.py --root-dir  /some/path  --remove_list list-file
It will then go through all .txt files under root-dir and edit appropriately
"""

import os
import argparse

def clean_tags(line, remove_set):
    # Split, strip, and deduplicate tags via set
    tags = set(t.strip() for t in line.split(",") if t.strip())
    newtags = tags - remove_set
    return ", ".join(sorted(newtags))

def process_file(filepath, remove_set):
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    new_lines = [clean_tags(line, remove_set) for line in lines]
    with open(filepath, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

def main(root_dir, remove_file):
    # Read remove-list
    with open(remove_file, "r", encoding="utf-8") as f:
        remove_set = set(line.strip() for line in f if line.strip())

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".txt"):
                filepath = os.path.join(dirpath, filename)
                process_file(filepath, remove_set)
                print(f"Processed: {filepath}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove tags from .txt files in a directory tree.")
    parser.add_argument("root_dir", help="Directory to walk")
    parser.add_argument("remove_list", help="Text file containing tags to remove (one per line)")
    args = parser.parse_args()
    main(args.root_dir, args.remove_list)

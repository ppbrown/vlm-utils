#!/usr/bin/env python3
import os
import argparse
from collections import Counter

def count_tags(root_dir):
    ctr = Counter()
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith('.txt'):
                path = os.path.join(dirpath, fname)
                with open(path, encoding='utf-8') as f:
                    line = f.read().strip()
                if not line:
                    continue
                tags = [tag.strip() for tag in line.split(',') if tag.strip()]
                ctr.update(tags)
    return ctr

def main():
    p = argparse.ArgumentParser(description="Count and sort AI caption tags in .txt files")
    p.add_argument('root', help="Root directory to search")
    args = p.parse_args()

    counts = count_tags(args.root)
    for tag, freq in counts.most_common():
        print(f"{tag}: {freq}")

if __name__ == "__main__":
    main()

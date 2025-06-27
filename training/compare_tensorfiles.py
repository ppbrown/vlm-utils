#!/usr/bin/env python3
"""
The purpose of this is to compare two saved tensor files, such as
create_t5_cache makes, to see how different they are.
Note that it can only take SINGLE-TENSOR(aka single token) files
"""


import argparse
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

def load_single_tensor(path):
    data = load_file(path, device="cpu")
    if len(data) != 1:
        raise ValueError(f"Expected exactly one tensor in {path}, found keys: {list(data.keys())}")
    return next(iter(data.values()))

def main():
    parser = argparse.ArgumentParser(
        description="Compute the cosine distance (1 - cosine similarity) between two tensors in safetensors files."
    )
    parser.add_argument("file1", help="First safetensors file")
    parser.add_argument("file2", help="Second safetensors file")
    args = parser.parse_args()

    t1 = load_single_tensor(args.file1).flatten()
    t2 = load_single_tensor(args.file2).flatten()

    sim = F.cosine_similarity(t1, t2, dim=0)
    distance = 1 - sim
    print(distance.item())

if __name__ == "__main__":
    main()

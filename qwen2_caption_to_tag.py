#!/bin/env python

"""
Purpose: given a directory, find all .txt files,
and summarize them as tagstyle output.
Write results to .txttags file
"""

import argparse
import glob
import multiprocessing as mp
import os
from pathlib import Path
from typing import Iterable, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ----------------- CONFIG -----------------
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
# ------------------------------------------

tokenizer = None
model = None


def init_worker(model_name: str, gpu_ids):
    """
    Runs once per worker process.
    Binds the process to a single GPU and loads the model there.
    """
    global tokenizer, model

    # Worker index in [0, len(gpu_ids)-1]
    worker_idx = mp.current_process()._identity[0] - 1
    gpu_id = gpu_ids[worker_idx]

    # Restrict this process to a single GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    #device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cuda"

    print(f"[worker {worker_idx}] Using GPU {gpu_id} -> {device}, loading model...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # 4-bit quantization to fit comfortably on P100 and improve throughput
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        load_in_4bit=True,
        device_map="auto",
        quantization_config=None,  # bitsandbytes default 4-bit config
        torch_dtype=torch.float16,
    )

    model.eval()
    if device == "cuda":
        torch.cuda.empty_cache()


def normalize_tags(text: str) -> str:
    """
    Convert model output into a clean 'tag1, tag2, tag3' line.
    """
    text = text.strip()
    # Replace various separators with commas
    for sep in ["\n", ";", "|", "\t"]:
        text = text.replace(sep, ",")

    # If the model gave something like "1. tag", strip numbering
    parts = []
    for raw in text.split(","):
        t = raw.strip()
        # Strip common list prefixes like "1." or "- "
        if t[:2].isdigit() and t[1:2] == ".":
            t = t[2:].strip()
        if t.startswith("- "):
            t = t[2:].strip()
        if t:
            parts.append(t.lower())

    # Deduplicate while preserving order
    seen = set()
    tags = []
    for t in parts:
        if t not in seen:
            seen.add(t)
            tags.append(t)

    return ", ".join(tags)


def caption_to_tags(caption: str) -> str:
    """
    Use the loaded LLM to convert a long caption into short tags.
    """
    global tokenizer, model

    # System + user instruction; tuned for tag-style output.
    messages = [
        {
            "role": "system",
            "content": (
                "You are a vision-tagging engine. "
                "Given a detailed natural-language image caption, "
                "you output a short, comma-separated list of tags "
                "(keywords and short phrases) describing the main visual "
                "objects, attributes, and style.\n\n"
                "Rules:\n"
                "- Only output tags, no full sentences.\n"
                "- Use plain English tags like 'woman', 'coffee cup', "
                "'city street', 'sunset', 'high contrast'.\n"
                "- Do NOT use anime tagging jargon such as '1girl', "
                "'masterpiece', 'best quality', '8k', 'nsfw', etc.\n"
                "- Prefer concrete visual concepts over abstract story details.\n"
                "- 5–25 tags is typical. "
                "Sort roughly from most to least important."
            ),
        },
        {
            "role": "user",
            "content": caption,
        },
    ]

    # For chat models like Qwen2.5-7B-Instruct
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    device = model.device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,          # deterministic for consistency
            temperature=0.0,
            top_p=1.0,
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Slice off the prompt tokens
    gen_ids = outputs[0, inputs["input_ids"].shape[1]:]
    raw_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    return normalize_tags(raw_text)


def process_one_file(path_str: str) -> Optional[str]:
    """
    Worker entry point: read caption, generate tags, write .txttags next to it.
    Returns the processed path for logging, or None on skip/error.
    """
    path = Path(path_str)

    # Skip our own outputs, just in case the glob catches them
    if str(path).endswith(".txttags"):
        return None

    out_path = Path(str(path) + "tags")  # foo.txt -> foo.txttags

    # Idempotent: skip if already processed
    if out_path.exists():
        return None

    try:
        caption = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not caption:
            return None

        tags = caption_to_tags(caption)
        if not tags:
            return None

        out_path.write_text(tags + "\n", encoding="utf-8")
        return str(path)
    except Exception as e:
        # Simple error logging; you can improve this for production
        print(f"[ERROR] {path}: {e}")
        return None


def iter_txt_files(root_dir: Path) -> Iterable[str]:
    """
    Lazily yield .txt files under root_dir (recursive).
    """
    pattern = str(root_dir / "**" / "*.txt")
    for path in glob.iglob(pattern, recursive=True):
        yield path


def main():
    parser = argparse.ArgumentParser(
        description="Bulk convert caption .txt files to tag-style .txttags using a local LLM."
    )
    parser.add_argument(
        "root",
        type=Path,
        help="Root directory containing caption .txt files (searched recursively).",
    )
    parser.add_argument(
        "--model",
        default=MODEL_NAME,
        help=f"Hugging Face model name (default: {MODEL_NAME})",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=None,
        help="Number of GPUs / workers to use (default: all visible GPUs).",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=4,
        help="multiprocessing imap_unordered chunksize (tune for throughput).",
    )

    args = parser.parse_args()

    root_dir = args.root
    if not root_dir.is_dir():
        raise SystemExit(f"{root_dir} is not a directory")

    # Detect GPUs (before workers muck with CUDA_VISIBLE_DEVICES)
    total_gpus = torch.cuda.device_count()
    if total_gpus == 0:
        raise SystemExit("No CUDA GPUs detected. P100s not visible?")

    n_workers = args.gpus or total_gpus
    if n_workers > total_gpus:
        n_workers = total_gpus

    gpu_ids = list(range(n_workers))
    print(f"Using {n_workers} worker processes on GPUs: {gpu_ids}")
    print(f"Model: {args.model}")
    print(f"Scanning for .txt files under {root_dir} ...")

    files_iter = iter_txt_files(root_dir)

    processed = 0
    with mp.Pool(
        processes=n_workers,
        initializer=init_worker,
        initargs=(args.model, gpu_ids),
    ) as pool:
        for result in pool.imap_unordered(process_one_file, files_iter, chunksize=args.chunksize):
            if result is not None:
                processed += 1
                if processed % 1000 == 0:
                    print(f"Processed {processed} files...")

    print(f"Done. Processed {processed} caption files.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()

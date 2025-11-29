#!/bin/env python

"""
Program to take a directory tree, find all the img files, and 
generate a .txtq caption file for them.
Default style is full sentences. However, can also output
somewhat of a "tag" style with --use_tags
Will automagically accomodate multi-GPU
"""

import argparse
import glob
import multiprocessing as mp
import os
from pathlib import Path
from typing import Iterable, Optional
from functools import partial

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"

processor = None
model = None
inference_prompt = None  # set per worker from CLI


def init_worker(model_name: str, gpu_ids, prompt_text: str):
    """
    Runs once per worker process.
    Binds the process to a single GPU and loads the VLM there.
    """
    global processor, model, inference_prompt

    # Worker index in [0, len(gpu_ids)-1]
    worker_idx = mp.current_process()._identity[0] - 1
    gpu_id = gpu_ids[worker_idx]

    # Restrict this process to a single GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    device = "cuda"

    print(f"[worker {worker_idx}] Using GPU {gpu_id} -> {device}, loading VLM...")

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    # 4-bit quantization to fit comfortably on P100 and improve throughput
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        trust_remote_code=True,
        load_in_4bit=True,        # bitsandbytes 4-bit quant
        device_map="auto",
        quantization_config=None, # default 4-bit config
        torch_dtype=torch.float16,
    )

    model.eval()
    if device == "cuda":
        torch.cuda.empty_cache()

    inference_prompt = prompt_text



def image_to_tags(image_path: Path, use_tags) -> str:
    """
    Use the loaded VLM to convert an image into short visual tags.
    """
    global processor, model, inference_prompt

    # Use an absolute file:// URI so Qwen2.5-VL can load it
    image_uri = "file://" + str(image_path.resolve())

    messages = [
        {
            "role": "system",
            "content": (
                "You are a vision-tagging engine. "
                "Given an image, you output a short, comma-separated list of tags "
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
            "content": [
                {"type": "image", "image": image_uri},
                {
                    "type": "text",
                    # <-- uses CLI-provided prompt text here
                    "text": inference_prompt,
                },
            ],
        },
    ]

    if not use_tags:
        messages[0]["content"] = "You are a helpful vision-language assistant."
    
    # Build chat prompt
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Let qwen_vl_utils handle image loading
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    device = model.device
    inputs = inputs.to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,   # deterministic for consistency
            temperature=0.0,
            top_p=1.0,
            num_beams=1,
        )

    # Slice off the prompt tokens
    in_len = inputs["input_ids"].shape[1]
    gen_ids = outputs[0, in_len:]

    raw_text = processor.batch_decode(
        gen_ids.unsqueeze(0),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return raw_text


def process_one_file(path_str: str, use_tags) -> Optional[str]:
    """
    Worker entry point: run VLM on image, generate tags,
    write foo.txtq next to it.
    Returns the processed path for logging, or None on skip/error.
    """
    path = Path(path_str)

    # Skip our own outputs, just in case
    if str(path).endswith(".txtq"):
        return None

    # image.jpg -> image.txtq
    out_path = path.with_suffix(".txtq")

    # Idempotent: skip if already processed
    if out_path.exists():
        return None

    try:
        tags = image_to_tags(path, use_tags)
        if not tags:
            return None

        out_path.write_text(tags + "\n", encoding="utf-8")
        return str(path)
    except Exception as e:
        # Simple error logging; you can improve this for production
        print(f"[ERROR] {path}: {e}")
        return None


def iter_image_files(root_dir: Path) -> Iterable[str]:
    """
    Lazily yield image files under root_dir (recursive).
    """
    exts = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp", "*.gif", "*.tif", "*.tiff")
    for ext in exts:
        pattern = str(root_dir / "**" / ext)
        for path in glob.iglob(pattern, recursive=True):
            yield path


def main():
    parser = argparse.ArgumentParser(
        description="Bulk generate .txtq files for images using a local Qwen2.5-VL VLM."
    )
    parser.add_argument(
        "root",
        type=Path,
        help="Root directory containing image files (searched recursively).",
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
    parser.add_argument(
        "--prompt",
        type=str,
        default="describe the image",
        help='Prompt text given to the VLM (default: "describe the image").',
    )
    parser.add_argument(
        "--use_tags",
        type=bool,
        default=False,
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
    print(f"Prompt: {args.prompt!r}")
    print(f"Scanning for image files under {root_dir} ...")

    files_iter = iter_image_files(root_dir)

    processed = 0
    with mp.Pool(
        processes=n_workers,
        initializer=init_worker,
        initargs=(args.model, gpu_ids, args.prompt),
    ) as pool:
        worker = partial(process_one_file, use_tags=args.use_tags)
        for result in pool.imap_unordered(worker, files_iter, chunksize=args.chunksize):
            if result is not None:
                processed += 1
                if processed % 1000 == 0:
                    print(f"Processed {processed} files...")

    print(f"Done. Processed {processed} image files.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()

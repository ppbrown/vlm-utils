#!/usr/bin/env python3
"""
build_t5cache_gpu.py
~~~~~~~~~~~~~~~~~~~
Generate T5-caption caches using your custom Diffusers pipeline.
Leaves existing cache alone  unless --overwrite is set

For every  caption.txt  under  --data_root
    caption.txt_t5cache   (bf16 safetensors) is written.

• Uses pipeline.encode_prompt()   (no manual tokeniser / encoder calls)
• Default model: opendiffusionai/stablediffusion_t5
• Runs entirely on a single GPU for max throughput
"""

import argparse, gc
from pathlib import Path

import torch, safetensors.torch as st
from diffusers import DiffusionPipeline
from tqdm import tqdm

#CACHE_POSTFIX="_t5_768"
# have to keep this until I update training backend
CACHE_POSTFIX="_t5cache"

# --------------------------------------------------------------------------- #
def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True,
                   help="directory tree that contains *.txt caption files")
    p.add_argument("--model",
                   default="/BLUE/t5-train/models/t5-sdx",
                   help="HF repo / local dir of your pipeline")
    #               default="opendiffusionai/stablediffusion_t5",

    p.add_argument("--batch_size", type=int, default=16,
                   help="Default=16.  (4096 => 1.3 GB @77 tokens)")
    p.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16",
                   help="GPU compute precision")
    p.add_argument("--overwrite", action="store_true",
                   help=f"re-encode even if *{CACHE_POSTFIX} already exists")
    return p.parse_args()

# --------------------------------------------------------------------------- #
@torch.inference_mode()
def encode_gpu(captions, pipe, precision):
    """
    Call pipeline.encode_prompt on GPU and return bf16 CPU tensor.
    """
    cast = torch.bfloat16 if precision == "bf16" else torch.float16
    with torch.autocast("cuda", cast):
        emb = pipe.encode_prompt(
            captions,
            device="cuda",                   # <-- tell helper to run on GPU
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )                                    # (B, T, 2048) on GPU
    return emb.to(torch.bfloat16, copy=False).cpu()

# --------------------------------------------------------------------------- #
def main():
    args = cli()
    torch.backends.cuda.matmul.allow_tf32 = True          # fastest safe path

    # ---------- load your pipeline ---------------------------------------- #
    print("Loading",args.model)
    pipe = DiffusionPipeline.from_pretrained(
        args.model,
        custom_pipeline=args.model,
        torch_dtype=torch.float16 if args.dtype == "fp16" else torch.bfloat16,
    )

    # Free modules unused during caption encoding to save VRAM
    for attr in ("vae", "unet", "image_encoder", "scheduler"):
        if hasattr(pipe, attr):
            setattr(pipe, attr, None)
 
    pipe.to("cuda")

    gc.collect(); torch.cuda.empty_cache()

    # ---------- gather caption files -------------------------------------- #
    root = Path(args.data_root).expanduser().resolve()
    txt_files = sorted(root.rglob("*.txt"))


    print("Parsing",root)
    def needs_cache(p: Path) -> bool:
        return args.overwrite or not (p.with_suffix(p.suffix + CACHE_POSTFIX)).exists()

    txt_files = [p for p in txt_files if needs_cache(p)]
    print(f"🟢 Encoding {len(txt_files):,} captions on GPU")

    # ---------- process in batches ---------------------------------------- #
    bs = args.batch_size
    for start in tqdm(range(0, len(txt_files), bs), unit="batch"):
        batch_files = txt_files[start : start + bs]
        captions = [p.read_text(encoding="utf-8").strip() for p in batch_files]

        emb = encode_gpu(captions, pipe, args.dtype)       # bf16 CPU tensor

        for path, vec in zip(batch_files, emb):
            st.save_file({"emb": vec}, path.with_suffix(path.suffix + CACHE_POSTFIX))

    print("✅ All caches written.")

if __name__ == "__main__":
    main()

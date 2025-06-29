#!/usr/bin/env python3

"""
Almost identical to create_img_cache.
Primary differences are:
    1. it has a default model, for sd1.5
    2. it has a different default output file
       This makes it feasible to have the same directory 
       contain latents for both sdxl and sd1.5
"""

import argparse
from pathlib import Path
from tqdm.auto import tqdm

import torch
import torchvision.transforms as TVT
import safetensors.torch as st
from diffusers import DiffusionPipeline
from PIL import Image

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5",
                   help="HF repo or local dir (must have a VAE)")
    p.add_argument("--data_root", required=True, help="Directory containing images (recursively searched)")
    p.add_argument("--out_suffix", default=".img_sdvae", help="File suffix for saved latents(default: _cache)")
    p.add_argument("--resolution", type=int, default=512, help="Resolution for images (default: 512)")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--extensions", nargs="+", default=["jpg", "jpeg", "png", "webp"])
    p.add_argument("--custom", action="store_true",help="Treat model as custom pipeline")
    return p.parse_args()

def find_images(input_dir, exts):
    images = []
    for ext in exts:
        images += list(Path(input_dir).rglob(f"*.{ext}"))
    return sorted(images)

def get_transform(size):
    return TVT.Compose([
        TVT.Lambda(lambda im: im.convert("RGB")),
        TVT.Resize(size, interpolation=Image.BICUBIC),
        TVT.CenterCrop(size),
        TVT.ToTensor(),
        # Have to do this before appplying VAE!!
        TVT.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

@torch.no_grad()
def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load pipeline (for VAE)
    pipe = DiffusionPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
        custom_pipeline=args.model if args.custom else None,
    )
    vae = pipe.vae.to(device)
    vae.eval()

    # Collect images
    image_paths = find_images(args.data_root, args.extensions)
    if not image_paths:
        print("No images found.")
        return

    tfm = get_transform(args.resolution)


    print(f"Processing {len(image_paths)} images from {args.data_root}")
    print(f"Using {args.model} to {args.out_suffix}...")
    print("")

    for i in tqdm(range(0, len(image_paths), args.batch_size)):
        batch_paths = image_paths[i:i+args.batch_size]
        batch_imgs = []
        valid_paths = []
        for path in batch_paths:
            try:
                img = Image.open(path)
                img = tfm(img)
                batch_imgs.append(img)
                valid_paths.append(path)
            except Exception as e:
                print(f"Could not load {path}: {e}")

        if not batch_imgs:
            continue

        batch_tensor = torch.stack(batch_imgs).to(device)
        latents = vae.encode(batch_tensor).latent_dist.sample().cpu()  # raw latent, no scaling

        # Save each latent as its own safetensors file
        for j, path in enumerate(valid_paths):
            out_path = path.with_name(path.stem + args.out_suffix)
            st.save_file({"latent": latents[j]}, str(out_path))

if __name__ == "__main__":
    main()

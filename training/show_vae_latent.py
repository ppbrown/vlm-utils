#!/bin/env python

# Pass in a latent image cachefile
# attemppt to decode and display it


import argparse
from pathlib import Path
import torch
import safetensors.torch as st
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image
from diffusers import DiffusionPipeline
from PIL import Image

# Check if CUDA is available and set the device accordingly
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

parser = argparse.ArgumentParser()
parser.add_argument("--model", help="Diffusers model directory or repo (must have VAE)",
                    default="/BLUE/t5-train/models/sd-base")
#                    default="/BLUE/t5-train/models/sdxl-orig")
parser.add_argument("--file", required=True, help="Path to a vae cache file")
parser.add_argument("--custom", action="store_true",help="Look for custom pipeline in the model")
args = parser.parse_args()

print(f"Using model {args.model} on file {args.file}")

pipe = DiffusionPipeline.from_pretrained(
    args.model,
    torch_dtype=torch.float32,
    safety_checker=None,
    requires_safety_checker=False,
    custom_pipeline=args.model if args.custom else None,
)
vae_model = pipe.vae.to(device).eval()

with torch.no_grad():
    cached = st.load_file(args.file)["latent"].to(device)
    #print(cached.shape)
    decoded_image = vae_model.decode(cached.unsqueeze(0)).sample

decoded_image = (decoded_image / 2 + 0.5).clamp(0, 1)  # Undo normalization
decoded_image = decoded_image.squeeze(0).cpu()          # Remove batch dimension

pil_image = to_pil_image(decoded_image)

pil_image.show()

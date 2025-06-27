import argparse
from pathlib import Path
import torch
import safetensors.torch as st
import torchvision.transforms as transforms
from diffusers import DiffusionPipeline
from PIL import Image

# Check if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--model", help="Diffusers model directory or repo (must have VAE)",
                    default="/BLUE/t5-train/models/sdxl-orig")
parser.add_argument("--file", required=True, help="Path to the latent .safetensors file")
parser.add_argument("--custom", action="store_true",help="Treat model as custom pipeline")
args = parser.parse_args()

print("Using model", args.model)

pipe = DiffusionPipeline.from_pretrained(
    args.model,
    torch_dtype=torch.float32,
    safety_checker=None,
    requires_safety_checker=False,
    custom_pipeline=args.model if args.custom else None,
)
vae_model = pipe.vae.to(device).eval()

input_image = Image.open(args.file).convert("RGB")
transform = transforms.Compose([
    transforms.Resize((512, 512)),  # Resize to 512x512 for consistency
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Coordinate transformation for VAE input. Mandatory!
])

input_tensor = transform(input_image).unsqueeze(0).to(device)  # Add batch dimension and move to device

# Encode the image
with torch.no_grad():
    encoded = vae_model.encode(input_tensor).latent_dist.sample()
    st.save_file({"latent": encoded}, "tempfile")

    cached = st.load_file("tempfile")["latent"].to(device)
    decoded_image = vae_model.decode(cached).sample

decoded_image = (decoded_image / 2 + 0.5).clamp(0, 1)  # Undo normalization
decoded_image = decoded_image.squeeze(0).cpu()          # Remove batch dimension

#to_pil = transforms.ToPILImage()
#pil_image = to_pil(decoded_image)
from torchvision.transforms.functional import to_pil_image

pil_image = to_pil_image(decoded_image)


pil_image.show()

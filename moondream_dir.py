#!/bin/env python

# Give this a directory name, and it will recursively process
# all image files under that directory
# Example custom prompts:
#   "Describe .... in the image."
#   "Is the image watermarked?." (mixed success)
#   "Are there humans in the image?"
#   "Is this a photograph?" (as opposed to a painting, etc)
#       It also sometimes identifies "this is a black and white photograph"

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import sys, os


model_id = "vikhyatk/moondream2"
revision = "05d640e6da70c37b2473e0db8fef0233c0709ce4" # Pin to specific version

model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, revision=revision,
    torch_dtype=torch.float16,
    #attn_implementation="flash_attention_2",
    device_map="cuda:0"
)
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

# use moondream built-in caption instead of custom prompt
def caption(image, calltype):
    answer = model.caption(image, calltype, stream=False)
    return answer.get("caption", "")


def process_image(image_path, call_type, prompt=None, suffix="moon"):
    filename, _ = os.path.splitext(image_path)
    txt_filename = f"{filename}.{suffix}"
    if os.path.exists(txt_filename):
        print(f"{txt_filename} already exists")
        return
    try:
        image = Image.open(image_path)
    except Exception as e:
        print(f"Failed to open {image_path}: {e}")
        return
    try:
        if call_type:
            response = caption(image, call_type)
        else:
            enc_image = model.encode_image(image)
            response = model.answer_question(enc_image, prompt, tokenizer)
    except Exception as e:
        print(f"Error processing {image_path}: Image corrupt somewhere", file=sys.stderr)
        exit(1)
    
    print(response)
    with open(txt_filename, "w") as f:
        f.write(response)

def main():
    parser = argparse.ArgumentParser(
        description="Recursively process all image files in a directory."
    )
    parser.add_argument("directory", help="Directory to process")
    parser.add_argument("-s", type=str, metavar="suffix", default="moon", help="default='moon'")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-c", action="store_true", help="Use short call")
    group.add_argument("-C", action="store_true", help="Use long call")
    group.add_argument("-p", type=str,  metavar="prompt", help="Custom prompt to use with answer_question")

    
    args = parser.parse_args()

    # Determine call_type based on the optional arguments.
    call_type = None
    PROMPT = None
    if args.c:
        call_type = "short"
    elif args.C:
        call_type = "normal"
    elif args.p:
        PROMPT = args.p

    # Walk through all subdirectories and files.
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    for root, dirs, files in os.walk(args.directory):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext not in IMAGE_EXTENSIONS:
                continue
            image_path = os.path.join(root, file)
            process_image(image_path, call_type, PROMPT, args.s)

if __name__ == '__main__':
    main()


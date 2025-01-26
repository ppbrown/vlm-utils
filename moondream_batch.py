#!/bin/env python

# This uses the 'moondream2' aka 2b model
#
# Give a bunch of image filenames on stdin.
# For example "image1.png"
# It will then write out a caption to "image1.txt"
#
# On a 4090 it can process 2-3 small images a second

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image

import sys, os

PROMPT = "Describe this image."

model_id = "vikhyatk/moondream2"
revision = "2024-08-26"  # Pin to specific version

model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, revision=revision,
    torch_dtype=torch.float16, attn_implementation="flash_attention_2",
    device_map="cuda:0"
)

tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
while True:
    try:
        image_path = input()
    except EOFError:
        exit()
    if image_path == '':
        exit()
    filename, _ = os.path.splitext(image_path)
    #txt_filename = f"{filename}.moon2"
    txt_filename = f"{filename}.txt"

    image = Image.open(image_path)
    enc_image = model.encode_image(image)
    response = model.answer_question(enc_image, PROMPT, tokenizer)
    print(response)
    with open(txt_filename, "w") as f:
        f.write(response)




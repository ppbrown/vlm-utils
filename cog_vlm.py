#!/usr/bin/env python

# script to use CogVLM to describe a list of files,
# given one at a time, on stdin
# Even with quant 4, this takes "12776MB" of vram.
# I dont know if that fits in 12GB cards

# Code borrowed from
# https://github.com/THUDM/CogVLM/basic_demo/

import argparse
import torch
import os

from PIL import Image
from transformers import AutoModelForCausalLM, LlamaTokenizer

query = "describe the image"
# alternative queries:
# query = "describe the style and content of this picture"
# query = "Should this image be rated nsfw or sfw?"


parser = argparse.ArgumentParser()
parser.add_argument("--quant", choices=[4], type=int, default=4, help='quantization bits')
# Yes there are other tuned versions of cog, used for general chat
# You dont want that. you want the VLM one.
#parser.add_argument("--from_pretrained", type=str, default="THUDM/cogagent-chat-hf", help='pretrained ckpt')
parser.add_argument("--from_pretrained", type=str, default="THUDM/cogvlm-grounding-generalist-hf", help='pretrained ckpt')
parser.add_argument("--local_tokenizer", type=str, default="lmsys/vicuna-7b-v1.5", help='tokenizer path')
parser.add_argument("--fp16", action="store_true")
parser.add_argument("--bf16", action="store_true")

args = parser.parse_args()
MODEL_PATH = args.from_pretrained
TOKENIZER_PATH = args.local_tokenizer
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = LlamaTokenizer.from_pretrained(TOKENIZER_PATH)
if args.bf16:
    torch_type = torch.bfloat16
else:
    torch_type = torch.float16

print("========Use torch type as:{} with device:{}========\n\n".format(torch_type, DEVICE))

if args.quant:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch_type,
        low_cpu_mem_usage=True,
        load_in_4bit=True,
        trust_remote_code=True
    ).eval()
else:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch_type,
        low_cpu_mem_usage=True,
        load_in_4bit=args.quant is not None,
        trust_remote_code=True
    ).to(DEVICE).eval()


while True:

    history = []
    print("Reading filenames from stdin now. Be patient...")

    while True:
        try:
            image_path = input()
        except EOFError:
            exit()

        if image_path == '':
            exit()

        filename, _ = os.path.splitext(image_path)
        txt_filename = f"{filename}.txt"
        if os.path.exists(txt_filename):
            print(txt_filename,"already exists")
            continue

        image = Image.open(image_path).convert('RGB')

        if image is None:
            exit()

        input_by_model = model.build_conversation_input_ids(
                tokenizer, query=query, history=history, images=[image])

        inputs = {
            'input_ids': input_by_model['input_ids'].unsqueeze(0).to(DEVICE),
            'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(DEVICE),
            'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(DEVICE),
            'images': [[input_by_model['images'][0].to(DEVICE).to(torch_type)]] if image is not None else None,
        }
        if 'cross_images' in input_by_model and input_by_model['cross_images']:
            inputs['cross_images'] = [[input_by_model['cross_images'][0].to(DEVICE).to(torch_type)]]

        # add any transformers params here.
        gen_kwargs = {"max_length": 2048,
                      "do_sample": False} # "temperature": 0.9
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            response = tokenizer.decode(outputs[0])
            response = response.split("</s>")[0]
            print(response)
            with open(txt_filename, "w") as f:
                f.write(response)
              

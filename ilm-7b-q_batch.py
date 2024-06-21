
# This has the 7billion-param model, quantized down to 4 bits
# Not only does it take less memory, but its a lot faster

import os
import torch, auto_gptq
from transformers import AutoModel, AutoTokenizer
from auto_gptq.modeling import BaseGPTQForCausalLM

auto_gptq.modeling._base.SUPPORTED_MODELS = ["internlm"]
torch.set_grad_enabled(False)

class InternLMXComposer2QForCausalLM(BaseGPTQForCausalLM):
    layers_block_name = "model.layers"
    outside_layer_modules = [
        'vit', 'vision_proj', 'model.tok_embeddings', 'model.norm', 'output',
    ]
    inside_layer_modules = [
        ["attention.wqkv.linear"],
        ["attention.wo.linear"],
        ["feed_forward.w1.linear", "feed_forward.w3.linear"],
        ["feed_forward.w2.linear"],
    ]

model = InternLMXComposer2QForCausalLM.from_quantized(
  'internlm/internlm-xcomposer2-vl-7b-4bit', trust_remote_code=True, device="cuda:0").eval()
tokenizer = AutoTokenizer.from_pretrained(
  'internlm/internlm-xcomposer2-vl-7b-4bit', trust_remote_code=True)

#query = '<ImageHere>Please describe this image in detail.'
query = '<ImageHere>Please objectively describe the subjects in detail, including any blurring.'

while True:
        try:
            image_path = input()
        except EOFError:
            exit()

        if image_path == '':
            exit()

        filename, _ = os.path.splitext(image_path)
        txt_filename = f"{filename}.ilm7q"

        if os.path.exists(txt_filename):
            print(txt_filename,"already exists")
            continue


        image = image_path

        with torch.cuda.amp.autocast():
            response, _ = model.chat(tokenizer, query=query, image=image, history=[], do_sample=False)
        print(response)
        with open(txt_filename, "w") as f:
            f.write(response)

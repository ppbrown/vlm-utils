#!/bin/env python

# This uses InternLM to process 5 512x512 images/second on a 4090
# Pass in a list of images on stdin
# Prints out only those with watermarks

import torch, os
from transformers import AutoModel, AutoTokenizer

torch.set_grad_enabled(False)
model = AutoModel.from_pretrained('internlm/internlm-xcomposer2-vl-1_8b',
        trust_remote_code=True).cuda().eval()
tokenizer = AutoTokenizer.from_pretrained('internlm/internlm-xcomposer2-vl-1_8b',
        trust_remote_code=True)

query = "<ImageHere> Does this image contain a watermark?"
while True:
        try:
            image_path = input()
        except EOFError:
            exit()

        if image_path == '':
            exit()

        filename, _ = os.path.splitext(image_path)
        image = image_path

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                response, _ = model.chat(tokenizer, query=query, image=image, history=[], do_sample=False)
        if response.strip().lower().startswith("yes"):
            print(image_path)

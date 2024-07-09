
"""
This code is inspired/modified from 
 https://github.com/haotian-liu/LLaVA

Run it and feed it a list of image filenames.

echo your_img.jpg | thisfile.py

with generate an autocaption file of your_img.txt

TUNABLES:
    "prompts"
    "txt_filename"
    (model settings)

For prompt, search for "prompts"
For model settings, jump to the end, then change 
model/4bit/8bit/device
settings as desired.
 (i usually makes copies of the script for convenience.
  eg: "llava-13b-batch.py")

Note particularly the 4bit and 8bit settings

(or technically you COULD actually override the model settings
  via the command-line options!)

"""




import argparse
import torch

import os

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer


def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

prompts = [ "describe the subjects in detail using objective language" ]

# if you arent in a hurry, you can use multiple.
# This gives you greater detail for each prompt, rather than
# putting everything into a single prompt.
"""
prompts = [
        "Describe the age, gender, hair type and hair color of the subjects.",
        "Describe the clothing in detail.",
        "Describe the direction the subjects are facing.",
        "Describe the direction the subjects are looking.",
        "Describe the ethnicity of subjects, using terminology such as Caucasian, African, Asian, etc."
        ]
"""


def main(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    args.conv_mode = conv_mode


    ## Loop here...
    while True:
        image_file=input()
        if not image_file:
            exit(0)
        filename, _ = os.path.splitext(image_file)
        txt_filename = f"{filename}.txt"
        if os.path.exists(txt_filename):
            print(txt_filename,"already exists")
            continue

        image = load_image(image_file)
        image_size = image.size
        image_tensor = process_images([image], image_processor, model.config)
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)

        conv = conv_templates[args.conv_mode].copy()
        if "mpt" in model_name.lower():
            roles = ('user', 'assistant')
        else:
            roles = conv.roles

        outtext=""
        for inp in prompts:
            #inp = input(f"{roles[0]}: ")

            if image is not None:
                # first message
                if model.config.mm_use_im_start_end:
                    inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
                else:
                    inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
                image = None
            
            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=[image_size],
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                    streamer=streamer,
                    use_cache=True)

            outputs = tokenizer.decode(output_ids[0]).strip()
            conv.messages[-1][-1] = outputs

            outputs = outputs.removesuffix('</s>').removeprefix('<s> ')+" "
            outtext += outputs

        # prompts...
        with open(txt_filename, "w") as f:
            f.write(outtext)

######################################################

##################################################################


parser = argparse.ArgumentParser()
"""
  known models:
  liuhaotian/llava-v1.6-34b
  liuhaotian/llava-v1.5-7b
  liuhaotian/llava-v1.5-13b
"""
parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.6-34b")
parser.add_argument("--model-base", type=str, default=None)
#parser.add_argument("--image-file", type=str, required=True)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--conv-mode", type=str, default=None)
parser.add_argument("--temperature", type=float, default=0.2)
parser.add_argument("--max-new-tokens", type=int, default=512)
parser.add_argument("--load-8bit", action="store_true")
parser.add_argument("--load-4bit", 
        action="store_true",default=True)
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()
main(args)

import torch, os
from transformers import AutoModel, AutoTokenizer

torch.set_grad_enabled(False)

#model_name='internlm/internlm-xcomposer2-vl-1_8b'
model_name='internlm/internlm-xcomposer2-4khd-7b'

model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16, trust_remote_code=True).cuda().eval()
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

query = '<ImageHere>Please describe this image in detail.'
##query = '<ImageHere>Please describe this image.'

while True:
        try:
            image_path = input()
        except EOFError:
            exit()

        if image_path == '':
            exit()

        filename, _ = os.path.splitext(image_path)
        txt_filename = f"{filename}.ilm7"


        image = image_path

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                response, _ = model.chat(tokenizer, query=query, image=image, history=[], do_sample=False)
        print(response)
        with open(txt_filename, "w") as f:
            f.write(response)

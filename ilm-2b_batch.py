
# util to take in image filenames in stdin, and write out caption files for each of them
# dir/somefile.jpg will get   dir/somefile.ilm written
# Change ".ilm" to ".txt" in the code if it is more convenient for you


import torch, os
from transformers import AutoModel, AutoTokenizer

torch.set_grad_enabled(False)

model = AutoModel.from_pretrained('internlm/internlm-xcomposer2-vl-1_8b',
        trust_remote_code=True).cuda().eval()
tokenizer = AutoTokenizer.from_pretrained('internlm/internlm-xcomposer2-vl-1_8b',
        trust_remote_code=True)

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
        txt_filename = f"{filename}.ilm"


        image = image_path

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                response, _ = model.chat(tokenizer, query=query, image=image, history=[], do_sample=False)
        print(response)
        with open(txt_filename, "w") as f:
            f.write(response)

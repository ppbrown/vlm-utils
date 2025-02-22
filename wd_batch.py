#!/bin/env python

# This code is a little different from the other "_batch" progs here.
# It does NOT take a list of files from stdin.
# Instead, it takes the name of a directory. It will then attempt to
# process all images in the directory using the "WD" tagger varient in 
# the named repo below.
# It will also REMOVE some of the tags the model typically outputs
# Adjust BLACKLIST as desired, along with the other tunables in this first section

# if you have multiple directories, and enough CPU/GPU/VRAM, you might be able
# to run multiple copies of this in parallel to double or triple your throughput
# On a 4090, a single execution can process around 7x 2mp images a second.
# 3 parallel instances will start hitting 100% GPU usage on peak



import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
import onnxruntime as rt # Make sure to install "onnxruntime-gpu" for acceleration!
import huggingface_hub
from tqdm import tqdm

MODEL_REPO = "SmilingWolf/wd-eva02-large-tagger-v3"
# MODEL_REPO = "SmilingWolf/wd-vit-large-tagger-v3"
#  (You can put any of the other SmilingWolf versions here)
MODEL_FILENAME = "model.onnx"
LABEL_FILENAME = "selected_tags.csv"
GENERAL_THRESHOLD = 0.50
CHARACTER_THRESHOLD = 0.50
MAX_TAGS = 60
BATCH_SIZE = 8
BLACKLIST = [
    "uncensored", "teeth","monster girl",
    "k-pop","asian","cosplay","photorealistic","photo background","realistic","solo",
    "multicolored hair","real world location","horror (theme)"
]

### END of tunable section


RATING_FORMAT = "rating:{}"
kaomojis = [
    "0_0","(o)_(o)","+_+","+_-","._.","<o>_<o>","<|>_<|>","=_=",">_<","3_3","6_9",">_o","@_@","^_^","o_o","u_u","x_x","|_|","||_||"
]

class Predictor:
    def __init__(self):
        self.model = None
        self.tag_names = None
        self.rating_indexes = None
        self.general_indexes = None
        self.character_indexes = None
        self.model_target_size = None
        self.load_model()

    def download_model(self):
        csv_path = huggingface_hub.hf_hub_download(MODEL_REPO, LABEL_FILENAME)
        model_path = huggingface_hub.hf_hub_download(MODEL_REPO, MODEL_FILENAME)
        return csv_path, model_path

    def load_model(self):
        csv_path, model_path = self.download_model()
        df = pd.read_csv(csv_path)
        name_series = df["name"].map(lambda x: x.replace("_", " ") if x not in kaomojis else x)
        self.tag_names = name_series.tolist()
        self.rating_indexes = list(np.where(df["category"] == 9)[0])
        self.general_indexes = list(np.where(df["category"] == 0)[0])
        self.character_indexes = list(np.where(df["category"] == 4)[0])
        try:
            self.model = rt.InferenceSession(model_path, providers=['TensorrtExecutionProvider','CUDAExecutionProvider','CPUExecutionProvider'])
        except:
            # self.model = rt.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            print("ERROR: executionprovider thing failed")
            exit(1)

        shape = self.model.get_inputs()[0].shape
        if len(shape) == 4:
            _, h, w, _ = shape
            self.model_target_size = h
        else:
            self.model_target_size = 448

    def prepare_image(self, image):
        t = self.model_target_size
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        c = Image.new("RGBA", image.size, (255, 255, 255))
        c.alpha_composite(image)
        image = c.convert('RGB')
        w, h = image.size
        m = max(w, h)
        l = (m - w)//2
        tp = (m - h)//2
        p = Image.new('RGB', (m, m), (255, 255, 255))
        p.paste(image, (l, tp))
        if m != t:
            p = p.resize((t, t), Image.BICUBIC)
        arr = np.asarray(p, dtype=np.float32)
        arr = arr[:, :, ::-1]
        return np.expand_dims(arr, 0)

    def predict_batch(self, images):
        a = [self.prepare_image(im) for im in images]
        b = np.vstack(a)
        i = self.model.get_inputs()[0].name
        o = self.model.get_outputs()[0].name
        preds = self.model.run([o], {i: b})[0]
        r = []
        for idx in range(len(images)):
            ls = list(zip(self.tag_names, preds[idx].astype(float)))
            rc = [ls[i] for i in self.rating_indexes]
            rating = max(rc, key=lambda x: x[1])[0]
            g = [ls[i] for i in self.general_indexes if ls[i][1] > GENERAL_THRESHOLD]
            c = [ls[i] for i in self.character_indexes if ls[i][1] > CHARACTER_THRESHOLD]
            r.append((rating, g, c))
        return r

def get_txt_path(img):
    b = os.path.splitext(os.path.basename(img))[0]
    return os.path.join(os.path.dirname(img), b + '.txt')


def process_new_images(predictor, images, save_metadata):
    for i in range(0, len(images), BATCH_SIZE):
        batch_files = images[i:i+BATCH_SIZE]
        imgs, paths = [], []
        for p in batch_files:
            try:
                im = Image.open(p)
                imgs.append(im)
                paths.append(p)
            except:
                continue
        if not imgs:
            continue
        try:
            res = predictor.predict_batch(imgs)
        except:
            continue
        for j, path in enumerate(paths):
            rating, g, c = res[j]
            g = [x for x in g if x[0] not in BLACKLIST]
            c = [("character:" + x[0], x[1]) for x in c]
            all_tags = g + c
            all_tags = sorted(all_tags, key=lambda x: x[1], reverse=True)
            all_tags = all_tags[:MAX_TAGS]
            tag_names = [x[0] for x in all_tags]
            out = RATING_FORMAT.format(rating) + ', ' + ', '.join(tag_names)
            with open(get_txt_path(path), 'w', encoding='utf-8') as f:
                f.write(out)

def process_directory(directory, save_metadata):
    if not os.path.exists(directory):
        return
    all_imgs = []
    for root, dirs, files in os.walk(directory):
        for f in files:
            if f.lower().endswith(('.jpg','.jpeg','.png','.bmp','.gif','.tiff','.webp')):
                all_imgs.append(os.path.join(root,f))
    if not all_imgs:
        return
    no_txt = []
    for img in all_imgs:
        if not os.path.exists(get_txt_path(img)):
            no_txt.append(img)
    p = Predictor()
    if no_txt:
        process_new_images(p, no_txt, save_metadata)

def main():
    if len(sys.argv)<2:
        print("You must specify a directory to process")
        sys.exit(1)
    directory = sys.argv[1]
    save_metadata = ("--metadata" in sys.argv)
    process_directory(directory, save_metadata)

if __name__ == '__main__':
    main()


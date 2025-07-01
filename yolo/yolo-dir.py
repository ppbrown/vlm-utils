#!/usr/bin/env python3

import sys
from pathlib import Path
from ultralytics import YOLO
from PIL import Image

# Config
CONF_LEVEL=0.25   #default is 0.25
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff")
ENGINE_PATH = "./yolov8x.engine"    # update path to your .engine file

def find_images(root):
    return [p for p in Path(root).rglob("*") if p.suffix.lower() in IMAGE_EXTS]

def save_txt(image_path, results):
    # Collect unique class names
    names = set()
    for box in results[0].boxes:
        cls_id = int(box.cls)
        # Prefer readable class name, fallback to id
        class_name = results[0].names[cls_id] if hasattr(results[0], 'names') else str(cls_id)
        names.add(class_name)
    # Write as comma-separated tag list, if not empty
    if names:
        txt_path = image_path.with_suffix('.txt')
        with open(txt_path, "w") as f:
            f.write(", ".join(sorted(names)))
            f.write("\n")

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} /path/to/images")
        sys.exit(1)
    root = sys.argv[1]
    model = YOLO(ENGINE_PATH)
    images = find_images(root)
    print(f"Found {len(images)} images.")
    for img_path in images:
        try:
            # catch corrupt files early
            #with Image.open(img_path) as im:
                #im.verify()
            results = model(str(img_path), conf=CONF_LEVEL, task="detect", verbose=False)
            save_txt(img_path, results)
            print(f"Processed: {img_path}")
        except Exception as e:
            print(f"Error: {img_path}: {e}")

if __name__ == "__main__":
    main()

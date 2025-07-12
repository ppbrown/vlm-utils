#!/bin/env python

import argparse
from pathlib import Path
from ultralytics import YOLO
import math

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}

def find_images(root):
    return [p for p in Path(root).rglob("*") if p.suffix.lower() in IMAGE_EXTS]

def main():
    parser = argparse.ArgumentParser(description="Batch watermark detection using YOLOv8.")
    parser.add_argument("dir", help="Root directory to search for images.")
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold (default 0.3)")
    parser.add_argument("--batch", type=int, default=64, help="Batch size (default 64)")
    args = parser.parse_args()

    model = YOLO("https://huggingface.co/qfisch/yolov8n-watermark-detection/resolve/main/best.pt")
    images = find_images(args.dir)
    print(f"Found {len(images)} images under {args.dir}")

    n = len(images)
    batch_size = args.batch

    for i in range(0, n, batch_size):
        batch_paths = images[i:i+batch_size]
        results = model.predict(
            source=[str(p) for p in batch_paths],
            imgsz=640,
            conf=args.conf,
            verbose=False,
            device=0,   # Explicitly use the first GPU
        )
        for path, res in zip(batch_paths, results):
            haswm = len(res.boxes) > 0
            status = "Watermark detected ✅" if haswm else "No watermark ❌"
            print(f"{path}: {status}")

if __name__ == "__main__":
    main()

#!/bin/env python

import os
import json
from PIL import Image
import sys

def check_image_dimensions(directory="."):
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(directory, filename)
            json_path = os.path.splitext(image_path)[0] + ".json"
            
            if not os.path.exists(json_path):
                continue
            
            try:
                with Image.open(image_path) as img:
                    actual_width, actual_height = img.size
                
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    expected_width = data.get("width")
                    expected_height = data.get("height")
                
                if (actual_width, actual_height) != (expected_width, expected_height):
                    print(image_path)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")

if __name__ == "__main__":
    directory = sys.argv[1] if len(sys.argv) > 1 else "."
    check_image_dimensions(directory)




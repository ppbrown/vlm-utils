#!/bin/bash

# parallel factor
PCOUNT=8
# resolution. Good choices are 336 or 512
export RES=512

# This requires ImageMagick to be installed.
# Give a source directory and a dest directory
# copy all .png or .jpg files from source to dest, resizing to square format of
# the specific size in the resize_image function
# This is for large scale AI captioning purposes, 
# where smaller images will get processed faster than larger ones.


# Ensure input and output directories are provided
if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Usage: $0 <input_directory> <output_directory>"
  exit 1
fi

INPUT_DIR="$1"
if [ ! -d "$INPUT_DIR" ]; then
  echo "Error: Directory '$INPUT_DIR' does not exist."
  exit 1
fi

OUTPUT_DIR="$2"
mkdir -p "$OUTPUT_DIR"

# Define the function for resizing
resize_image() {
  INPUT_FILE="$1"
  OUTPUT_FILE="$OUTPUT_DIR/$(basename "$INPUT_FILE")"
  convert "$INPUT_FILE" -resize ${RES}x${RES}\! "$OUTPUT_FILE"
  echo "Resized: $OUTPUT_FILE"
}

export -f resize_image
export OUTPUT_DIR

# Find all PNG files and process them in parallel 
find "$INPUT_DIR" -type f -name "*.png" -o -name "*.jpg" | parallel -j${PCOUNT} resize_image {}

echo "All IMG files resized in parallel and saved to $OUTPUT_DIR."


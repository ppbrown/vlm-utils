#  Purpose

One or more utils for various VLMs (Visual Language Models),
primarily around AI captioning

## Captioner of choice

I like the "moondream 2b" model, as of 2025, because it is
* Small
* Fast  
* Fairly accurate

So see [moondream_batch.py](moondream_batch.py) and also
[moondream_requirements.txt](moondream_requirements.txt)

## Benefits of downsizing

Note that AI captioning should run faster on small images than larger images... and except for some teeny-tiny
watermarks, it should see all it needs to see in a 512x512 version of some 3188x4096 monster.
So if you are doing a LARGE caption job, you may save some time by bulk conversion of the images. See also

[resize_imgs.sh](resize_imgs.sh)

It has a (adjustable) parallel factor to it, so you should be able to downsize 10 - 50 images per second on a super fast box, compared to caption tools that may take 1 second per 512px image, or 3 seconds on a 2048px image


# Older ones

Warning: most of these no longer work, because the model authors keep CHANGING THE API! Grrr.
If you want to submit a PR for a working update of any of them feel free to do so

## ilm-7b-q_batch.py

Similar to others, but takes less than 12 GB of vram, and is also the fastest to run

Includes a commented-out prompt to focus on detecting text in the image, which is good to find watermarks, etc.
In "find the text" mode, a 4090 processes around 1 image a second.

## ilm-7b_batch.py

Use it like cog_vlm.py below. 
Note that it generates ".ilm7" files instad of ".txt" files
Also, it requires 24GB rather than 12 GB of vram


## ilm-2b_batch.py

Use it like cog_vlm.py below. 
Note that it generates ".ilm" files instad of ".txt" files
Also, it requires 16GB rather than 12 GB of vram
Not sure why anyone would want to use this version instead of the 7gb quantized, but if you would like to try it out,
here is how you can do so.

## cog_vlm.py 

Note that you must have **at least** 12 GB vram to use

### cog_vlm.py purpose

This can be used for multiple related purposes:

* Autogenerating .txt descriptions for images
* Detecting if there is an artists signature or watermark in an image
* Determining if an image has nsfw content

  See the script comments for alternative prompts that may help

## llava-batch.py

This can be configured to use either the 7b, 13b, or 32b LLAVA model
Read the comments at the top of the file for details. This is slow, but potentially generates the
most detailed and accurate output, with the 32b model.

# Install

    python -m venv venv
    . venv/bin/activate
    pip install -r requirements.txt
    #(or the .txt file matching the script you want to use)

## Run

    python cog_vlm.py

### Typical Use

    # The program expects to read in filenames, one per line,
    # from stdin. This is a quick way to make that work

    ls *.jpg  .....  | col |python cog_vlm.py

    # Note that the first run will take a VERY LONG TIME, because
    # it will download a buncha stuff from hugging face.
    # but after that, it should start up after a few seconds.
    # There is still a loading penalty, however, so it is
    # most efficient to do more than one file at a time

## Filtering tip
It can be very useful for filtering images.

If you want to do this in a directory that already has some ".txt" files,
then modify the script to output to ".desc" instead of ".txt"

eg:   txt_filename = f"{filename}.desc" 

Once you have done this, then you can generate a list of image files
to potentially remove, with something like:

egrep -l 'watermark|artist.s signature|comic panel|blahblah' *desc |sed s/desc/jpg/

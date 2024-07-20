#  Purpose


One or more utils for various VLMs (Visual Language Models)

# ilm-7b-q_batch.py

Similar to others, but takes less than 12 GB of vram, and is also the fastest to run

Includes a commented-out prompt to focus on detecting text in the image, which is good to find watermarks, etc.

# ilm-2b_batch.py

Use it like cog_vlm.py below. 
Note that it generates ".ilm" files instad of ".txt" files
Also, it requires 16GB rather than 12 GB of vram

# ilm-7b_batch.py

Use it like cog_vlm.py below. 
Note that it generates ".ilm7" files instad of ".txt" files
Also, it requires 24GB rather than 12 GB of vram


# cog_vlm.py 

Note that you must have **at least** 12 GB vram to use

## cog_vlm.py purpose

This can be used for multiple related purposes:

* Autogenerating .txt descriptions for images
* Detecting if there is an artists signature or watermark in an image
* Determining if an image has nsfw content

  See the script comments for alternative prompts that may help

# llava-batch.py

This can be configured to use either the 7b, 13b, or 32b LLAVA model
Read the comments at the top of the file for details

## Install

    python -m venv venv
    . venv/bin/activate
    pip install -r requirements.txt

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

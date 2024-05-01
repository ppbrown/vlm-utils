# cogvlm-utils
One or more utils for the CogVLM Visual Language Model

Currently, it has only ONE script: cog_vlm.py
Note that you must have **at least** 12 GB vram to use

## Install

    python -m venv venv
    . venv/bin/activate
    pip install -r requirements.txt

## Run

    python cog_vlm.py

### Actual

    # The program expects to read in filenames, one per line,
    # from stdin. This is a quick way to make that work

    ls somefile.jpg  .....  | col |python cog_vlm.py (correct flags here as mentioned above)

    # Note that the first run will take a VERY LONG TIME, because
    # it will download a buncha stuff from hugging face.
    # but after that, it should start up after a few seconds.
    # There is still loading penalty, however, so it is
    # is most efficient to do more than one file at a time

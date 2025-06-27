# Training subdirectory

This directory holds my current attempts at a custom AI model training script,
along with associated utilities and cache creation tools.

I am using the CLI tools in an attempt to train a model from scratch, that is:

* SDXL vae
* T5 xxl text encoder
* SD 1.5 vae

I'm training this in bf16 precision and 512x512 images, because with this setup,
I can run a native batch size of 64

## Assumptions

Certain parts of these scripts are hard-coded around my model.
They should be easily adaptable to normal SD1.5 training.
Less easy for SDXL.
More difficult for any other model.

They do use the "diffusers" Pipeline methodology though.

The training script is a work in progress. Not guaranteed to work correctly at this point!

## Data prep

To use the training stuff, you need to prepare a dataset.
Initially, it should be a directory, or directory tree, with a bunch of image files
(usually .jpg) and a set of matching .txt files which contain a caption for its jpg twin.

## Cache generation

* image caching script (create_img_cache.py)
* text caption caching script (create_t5cache_768.py)


Note that both of these expepct to make use of the custom "diffusers pipeline" present in
huggingface model "opendiffusionai/stablediffusion_t5"
or "opendiffusionai/sdx_t5"

Sample usage;

    ./create_img_cache.py --model opendiffusionai/stablediffusion_t5 --data_root /data


## Training

You should only need to call the front end, "t5lion_caching.sh".

It takes care of invoking train_lion_caching.py

edit the front end to tweak the various flags used, such as

    --learning_rate 
    --batch_size


As noted above, I can just barely fit in a batch size of 64, on my 4090, using 512x512 resolution.
(in this case represented as --resolution 512)








# Training subdirectory

This directory holds my current attempts at a custom AI model training script,
along with associated utilities and cache creation tools.

I am using the CLI tools in an attempt to train a model from scratch, that is:

* SDXL vae
* T5 xxl text encoder
* SD 1.5 vae

I'm training this in bf16 precision and 512x512 images, because with this setup,
I can run a native batch size of 64 on my rtx 4090, and get
1600 steps per hour

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

    ./create_img_cache.py --model opendiffusionai/stablediffusion_t5 --data_root /data --custom


## Training

You should only need to call the front end, "t5lion_caching.sh".

It takes care of invoking train_lion_caching.py

edit the front end to tweak the various flags used, such as

    --learning_rate 
    --batch_size


As noted above, I can just barely fit in a batch size of 64, on my 4090, using 512x512 resolution.
(in this case represented as --resolution 512)

## Benefits

Benefits of this method over larger programs:

* You can easily identify the cache files. 
* You can also easily choose to regenerate JUST img cache or text cache files


# Square image limitation

By default, these tools will only work with 512x512 resolution.

Resolution is controlled by the size of the generated image cache files.

In theory, if you use the --resolution tweaking 
(and remove the CenterCrop call) in create_img_cache.py,
you could also train on other sizes. But BEWARE!

There is a theoretical upper limit on total amount of knowledge you can train.

SD1.5 is a relatively tiny model, and training at different sizes effectively loses knowledge.
If you want a model knowledable about many things, you must stick to one aspect ratio. 

Contrariwise, the more varients of size you train on for a particular subject, the more
you will displace knowledge about other things you are not training on.

# ------------------------------------------------------------------------

# Tensorboard logging

These scripts output logging to tensorboard.

With other programs, you may be used to seeing the typical "learning rate" and "loss"
graphs. This, however, adds in "qk_grads_av" and "raw loss".

This is because when you are training from scratch, it is really important to make sure
that the "q/k gradients" arent doing crazy things like going to 0.
It is also sometimes nice, when SNR is enabled, to compare the default loss stats, vs the
"raw (non snr)" loss


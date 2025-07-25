
# This is a front-end wrapper for training settings.
# Save different versions as your "config files"

####################################################
#
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1

LR=6.1e-5
BATCH=40
ACCUM=1
#SCHED=linear
SCHED=constant
GAMMA=4

DATADIR="/BLUE/CC8M/CC8M-squarish-simple"
MODEL=/BLUE/t5-train/models/t5-sdx  
OUTPUTDIR=./t5_sdx_simplecached_b${BATCH}a${ACCUM}_${LR}g${GAMMA} 
#OUTPUTDIR=./t5_sdx_simple_b${BATCH}a${ACCUM}_${LR}nosnr


# Note that this training script uses cached latents.
# Resolution is set in the img latent caches
accelerate launch train_lion_caching.py \
  --pretrained_model  $MODEL  \
  --train_data_dir    $DATADIR \
  --output_dir        $OUTPUTDIR \
  --batch_size        $BATCH \
  --gradient_accum    $ACCUM \
  --max_steps         100_000 \
  --save_steps        500 \
  --warmup_steps      0 \
  --learning_rate     $LR \
  --noise_gamma       $GAMMA \
  --scheduler         $SCHED \
  --gradient_checkpointing \
  --use_snr \
  --cpu_offload \
  --sample_prompt    "woman"


#  --sample_steps      10 \
#  --reinit_unet \


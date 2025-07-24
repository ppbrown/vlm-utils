
# This is a front-end wrapper for training settings.
# Edit as needed. 
# Save different versions of this file, as your "config files"
# Note that gets copied into the top level OUTPUTDIR
# automatically

####################################################
#
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1

LR=3e-5
BATCH=64
ACCUM=4

#SCHED=linear
SCHED=constant
#SCHED=cosine
#SCHED=cosine_with_min_lr

#GAMMA=5
STEPS=50000
WARMUP=0

DATADIR="${DATADIR} /ANKER/LAION/LAION-23-womanonly-cancrop"
#DATADIR="${DATADIR} /ANKER/LAION/LAION-AE-square-womansolo"


#MODEL=opendiffusionai/xllsd-alpha0
MODEL="stable-diffusion-v1-5/stable-diffusion-v1-5"

OUTPUTDIR="sd_test_b${BATCH}a${ACCUM}_${LR}_${SCHED}${STEPS}"

if [[ "$GAMMA" != "" ]] ; then
	USE_SNR="--use_snr --noise_gamma $GAMMA"
	OUTPUTDIR="$OUTPUTDIR"g${GAMMA}
fi

# Note that this training script uses cached latents.
# Resolution is set in the img latent caches
accelerate launch train_lion_caching.py \
  --pretrained_model  $MODEL  \
  --imgcache_suffix   ".img_sdvae" \
  --txtcache_suffix   ".txt_clipl" \
  --optimizer         lion \
  --output_dir        $OUTPUTDIR \
  --copy_config       $0 \
  --batch_size        $BATCH \
  --gradient_accum    $ACCUM \
  --max_steps         $STEPS \
  --save_on_epoch      \
  --warmup_steps      $WARMUP \
  --learning_rate     $LR \
  --scheduler         $SCHED \
  --gradient_checkpointing   \
  --cpu_offload              \
  $USE_SNR \
  --sample_prompt    "woman" "a beautiful woman" \
  --train_data_dir    $DATADIR 


#  --use_snr \
#  --sample_steps      10 \
#  --reinit_unet \


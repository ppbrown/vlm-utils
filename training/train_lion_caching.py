#!/usr/bin/env python

# Okay this is called "train_lion_" but you can actually override optimizer.
# Currently it only supports
#  --optimizer  adamw8

import argparse, os, math
from pathlib import Path
from tqdm.auto import tqdm

import torch
import safetensors.torch as st
from torch.utils.data import Dataset, DataLoader

from accelerate import Accelerator, DistributedDataParallelKwargs
from diffusers import DiffusionPipeline, UNet2DConditionModel
from diffusers import DDPMScheduler
from diffusers.models.attention import Attention as CrossAttention
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_snr

from torch.utils.tensorboard import SummaryWriter

import lion_pytorch

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

# --------------------------------------------------------------------------- #
# 1. CLI                                                                      #
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained_model", required=True,  help="HF repo or local dir")
    p.add_argument("--train_data_dir",  nargs="+", required=True,  help="directory tree(s) containing *.jpg + *.txt")
    p.add_argument("--optimizer",      type=str, choices=["adamw8","lion"], default="adamw8")
    p.add_argument("--copy_config",    type=str, help="config file to archive with training, if model load succeeds")
    p.add_argument("--output_dir",     required=True)
    p.add_argument("--batch_size",     type=int, default=4)
    p.add_argument("--gradient_accum", type=int, default=1, help="default=1")
    p.add_argument('--gradient_checkpointing', action='store_true',
                   help="enable grad checkpointing in unet")
    p.add_argument("--learning_rate",   type=float, default=1e-5, help="default=1e-5")
    p.add_argument("--weight_decay",   type=float)
    p.add_argument("--vae_scaling_factor", type=float, help="override vae scaling factor")
    p.add_argument("--text_scaling_factor", type=float, help="Override embedding scaling factor")
    p.add_argument("--learning_rate_decay", type=float,
                   help="Subtract this every epoch, if schedler==constant")
    p.add_argument("--max_steps",       type=int, default=10_000, help="default=10_000")
    ex_group = p.add_mutually_exclusive_group()
    ex_group.add_argument("--save_steps",    type=int)
    ex_group.add_argument("--save_on_epoch", action="store_true")
    p.add_argument("--warmup_steps",    type=int, default=0, help="default=0")
    p.add_argument("--noise_gamma",     type=float, default=5.0)
    p.add_argument("--cpu_offload", action="store_true",
                   help="enable cpu offload at pipe level")
    p.add_argument("--use_snr", action="store_true",
                   help="Use Min SNR noise adjustments")
    p.add_argument("--reinit_crossattn", action="store_true",
                   help="Attempt to reset cross attention weights for text realign")
    p.add_argument("--reinit_attn", action="store_true",
                   help="Attempt to reset ALL attention weights for text realign")
    p.add_argument("--reinit_qk", action="store_true",
                   help="Attempt to reset just qk weights for text realign")
    p.add_argument("--reinit_unet", action="store_true",
                   help="Train from scratch unet (Do not use, this is broken)")
    p.add_argument("--sample_prompt",  type=str, help="prompt to use for a checkpoint sample image")
    p.add_argument("--scheduler", type=str, default="constant", help="default=constant")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--txtcache_suffix", type=str, default=".txt_t5cache", help="default=.txt_t5cache")
    p.add_argument("--imgcache_suffix", type=str, default=".img_sdvae", help="default=.img_sdvae")

    return p.parse_args()

# --------------------------------------------------------------------------- #
# 2. Dataset                                                                  #
# --------------------------------------------------------------------------- #

class CaptionImgDataset(Dataset):
    """Iterate .jpg and .png files, but yield only cache files for both img and text."""
    def __init__(self, root_dirs, imgcache_suffix=".img_cache", txtcache_suffix=".txt_t5cache"):
        self.files = []
        extset = ("jpg", "png")
        for root in root_dirs:
            print(f"Scanning {root} for {imgcache_suffix} and {txtcache_suffix} matching {extset}")
            subtotal=0
            for ext in extset:
                for p in Path(root).rglob(f"*.{ext}"):
                    img_cache = p.with_suffix(imgcache_suffix)
                    txt_cache = p.with_suffix(txtcache_suffix)
                    # Only keep samples where BOTH caches exist
                    if img_cache.exists() and txt_cache.exists():
                        self.files.append((img_cache, txt_cache))
                        subtotal+=1
            print(f"Cache pairs found: {subtotal}")
        print(f"Total cache pairs found: {len(self.files)}")

        if not self.files:
            raise RuntimeError("No valid cache pairs found! Did you run your cache pre-processing script?")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_cache, txt_cache = self.files[idx]
        return {
            "img_cache": str(img_cache),
            "txt_cache": str(txt_cache),
        }

def collate_fn(examples):
    return {
        "img_cache": [e["img_cache"] for e in examples],
        "txt_cache": [e["txt_cache"] for e in examples],
    }

from diffusers.utils import logging as hf_logging


# PIPELINE_CODE_DIR is typicaly the dir of original model
def sample_img(prompt, seed, CHECKPOINT_DIR, PIPELINE_CODE_DIR, fname="sample.png"):
    outname=f"{CHECKPOINT_DIR}/{fname}"
    tqdm.write(f"Trying render of '{prompt}' using seed {seed} to {outname}...")
    hf_logging.disable_progress_bar()
    pipe = DiffusionPipeline.from_pretrained(
        CHECKPOINT_DIR, custom_pipeline=PIPELINE_CODE_DIR, use_safetensors=True,
        safety_checker=None, requires_safety_checker=False,
        torch_dtype=torch.bfloat16,
    )
    pipe.safety_checker=None

    pipe.enable_sequential_cpu_offload()
    generator = torch.Generator(device="cuda").manual_seed(seed)

    images = pipe(prompt, num_inference_steps=30, generator=generator).images
    images[0].save(outname)
    tqdm.write("Sample saved.")

    hf_logging.enable_progress_bar()

#####################################################
# Main                                              #
#####################################################

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    peak_lr       = args.learning_rate
    warmup_steps  = args.warmup_steps
    total_steps   = args.max_steps

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accum,
        mixed_precision="bf16" if torch.cuda.is_available() else "no",
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False)]
    )
    device = accelerator.device
    torch_dtype=torch.bfloat16 if accelerator.mixed_precision=="bf16" else torch.float32

    # ----- load pipeline --------------------------------------------------- #
    pipe = DiffusionPipeline.from_pretrained(
        args.pretrained_model,
        custom_pipeline=args.pretrained_model,
        torch_dtype=torch_dtype
    )

    if args.reinit_unet:
        print("Training Unet from scratch")
        """ This does not work!!
        BASEUNET="models/sd-base/unet"
        # Note: the config from pipe.unet seems to get corrupted.
        # SO, Load a fresh one instead
        conf=UNet2DConditionModel.load_config(BASEUNET)
        new_unet=UNet2DConditionModel.from_config(conf)
        print("UNet cross_attention_dim:", new_unet.config.cross_attention_dim)
        new_unet.to(torch_dtype)
        pipe.unet=new_unet
        """
        print("Attempting to reset ALL layers of Unet")
        from reinit import reinit_all_unet
        reinit_all_unet(pipe.unet)
    elif args.reinit_qk:
        print("Attempting to reset Q/K layers of Unet")
        from reinit import reinit_qk
        reinit_qk(pipe.unet)
    elif args.reinit_crossattn:
        print("Attempting to reset Cross Attn layers of Unet")
        from reinit import reinit_cross_attention
        reinit_cross_attention(pipe.unet)
    elif args.reinit_attn:
        print("Attempting to reset Attn layers of Unet")
        from reinit import reinit_all_attention
        reinit_all_attention(pipe.unet)
    else:
        print("Finetuning prior Unet")

    if args.cpu_offload:
        print("Enabling cpu offload")
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)

    if args.gradient_checkpointing:
        print("Enabling gradient checkpointing in UNet")
        pipe.unet.enable_gradient_checkpointing()

    vae, unet = pipe.vae.eval(), pipe.unet
    print("Overriding noise sched to DDPM")
    noise_sched = DDPMScheduler(
            num_train_timesteps=1000, # DIFFERENT from lr_sched num_training_steps
            # beta_schedule="cosine", # "cosine not implemented for DDPMScheduler"
            clip_sample=False
            )
    print("T5 (projection layer) scaling factor is", pipe.t5_projection.config.scaling_factor)

    # Freeze VAE (and T5) so only UNet is optimised; comment-out to train all.
    for p in vae.parameters():                p.requires_grad_(False)
    for p in pipe.text_encoder.parameters():  p.requires_grad_(False)
    for p in pipe.t5_projection.parameters(): p.requires_grad_(False)

    # ----- data ------------------------------------------------------------ #
    ds = CaptionImgDataset(args.train_data_dir, 
                           txtcache_suffix=args.txtcache_suffix,
                           imgcache_suffix=args.imgcache_suffix
                           )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=8, persistent_workers=True,
                    pin_memory=True, collate_fn=collate_fn,
                    prefetch_factor=4)

    # Gather just-trainable parameters
    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    if args.optimizer == "lion":
        import lion_pytorch
        # lion doesnt use decay?
        weight_decay=0.00
        optim = lion_pytorch.Lion(trainable_params, lr=peak_lr, weight_decay=0.00, betas=(0.95,0.98))
        #optim = lion_pytorch.Lion(trainable_params, lr=peak_lr, weight_decay=0.00, betas=(0.93,0.95))
    elif args.optimizer == "adamw8":
        import bitsandbytes as bnb
        if args.weight_decay:
            weight_decay=args.weight_decay
        else:
            weight_decay=0.01
        optim = bnb.optim.AdamW8bit(trainable_params, weight_decay=weight_decay, lr=peak_lr, betas=(0.95,0.98))
    else:
        print("ERROR: unrecognized optimizer setting")
        exit(1)

    print("Using optimizer",args.optimizer,"weight decay:",weight_decay)
    print(f"NOTE: peak_lr = {peak_lr}, lr_scheduler={args.scheduler}, batch={args.batch_size}, steps={total_steps}")
    for p in unet.parameters(): p.requires_grad_(True)
    unet, dl, optim = accelerator.prepare(pipe.unet, dl, optim)
    unet.train()

    lr_sched = get_scheduler(
        args.scheduler,
        optimizer=optim,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    lr_sched = accelerator.prepare(lr_sched)
    print(
        f"Align-phase: {sum(p.numel() for p in trainable_params)/1e6:.2f} M "
        "parameters will be updated"
    )

    latent_scaling = vae.config.scaling_factor

    global_step    = 0
    run_name = os.path.basename(args.output_dir)
    tb_writer = SummaryWriter(log_dir=os.path.join("tensorboard/",run_name))

    def checkpointandsave():
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{global_step:05}")
        pinned_te, pinned_unet = pipe.text_encoder, pipe.unet
        pipe.unet = accelerator.unwrap_model(unet)
        pipe.save_pretrained(ckpt_dir, safe_serialization=True)
        pipe.text_encoder, pipe.unet = pinned_te, pinned_unet
        if args.sample_prompt is not None:
            sample_img(args.sample_prompt, 99, ckpt_dir, 
                       args.pretrained_model, 
                       fname=f"../sample-s{global_step:05d}.png")
            if global_step == 0:
                # For step 0, treat chpt_dir as a neccessary temp save, just for
                # sample generation
                import shutil
                shutil.rmtree(ckpt_dir)
                if args.copy_config:
                    print("archiving", args.copy_config, "\n") # yes this needs extra LF
                    shutil.copy(args.copy_config, args.output_dir)

    # ----- training loop --------------------------------------------------- #
    ebar = tqdm(range(math.ceil(args.max_steps / len(dl))), 
                desc="Epoch", unit="", dynamic_ncols=True,
                position=0)
    for epoch in ebar:
        if args.save_on_epoch:
            checkpointandsave()

        pbar = tqdm(dl, 
                    desc="LocalStep", 
                    bar_format="{l_bar}{bar}|{n_fmt}/{total_fmt}{rate_fmt}{postfix}", 
                    dynamic_ncols=True,
                    position=1,
                    leave=False)

        for batch in pbar:
            with accelerator.accumulate(unet):
                # --- Load latents & prompt embeddings from cache ---
                latents = []
                for cache_file in batch["img_cache"]:
                    latent = st.load_file(cache_file)["latent"]
                    latents.append(latent)
                latents = torch.stack(latents).to(device, dtype=torch_dtype) * latent_scaling

                embeds = []
                for cache_file in batch["txt_cache"]:
                    emb = st.load_file(cache_file)["emb"]
                    emb = emb.to(device, dtype=torch_dtype)
                    embeds.append(emb)
                prompt_emb = torch.stack(embeds).to(device, dtype=torch_dtype)

                # --- Add noise ---
                noise = torch.randn_like(latents)
                bsz   = latents.size(0)
                timesteps = torch.randint(
                    0, noise_sched.config.num_train_timesteps,
                    (bsz,), device=device, dtype=torch.long
                )
                noisy_latents = noise_sched.add_noise(latents, noise, timesteps)

                # --- UNet forward & loss ---
                model_pred = unet(noisy_latents, timesteps,
                                  encoder_hidden_states=prompt_emb).sample

                mse = torch.nn.functional.mse_loss(
                        model_pred.float(), noise.float(), reduction="none")
                mse = mse.view(mse.size(0), -1).mean(dim=1)
                raw_mse_loss = mse.mean()

                if args.use_snr:
                    snr = compute_snr(noise_sched, timesteps)
                    gamma = args.noise_gamma
                    gamma_tensor = torch.full_like(snr, gamma)
                    weights = torch.minimum(snr, gamma_tensor) / (snr + 1e-8)
                    loss = (weights * mse).mean()
                else:
                    loss = raw_mse_loss

                accelerator.wait_for_everyone()
                accelerator.backward(loss)

            # -----logging & ckp save  ----------------------------------------- #
            if accelerator.is_main_process:
                qk_grad_sum = sum(
                        p.grad.abs().mean().item()
                        for n,p in unet.named_parameters()
                        if p.grad is not None and (".to_q" in n or ".to_k" in n))
                total_norm = 0.0
                for p in unet.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                for n, p in unet.named_parameters():
                    if p.grad is not None and torch.isnan(p.grad).any():
                        print(f"NaN grad: {n}")

                current_lr = lr_sched.get_last_lr()[0]
                pbar.set_postfix({"l": f"{loss.item():.3f}",
                                  "raw": f"{raw_mse_loss.item():.3f}",
                                  "qk": f"{qk_grad_sum:.2e}",
                                  "g": f"{total_norm:.2e}",
                                  "E": f"{epoch}",
                                  #"lr": f"{current_lr:.1e}",
                                  })

                if tb_writer is not None:
                    tb_writer.add_scalar("train/loss", loss.item(), global_step)
                    tb_writer.add_scalar("train/loss_raw", raw_mse_loss.item(), global_step)
                    tb_writer.add_scalar("train/learning_rate", current_lr, global_step)
                    tb_writer.add_scalar("train/qk_grads_av", qk_grad_sum, global_step)

            if (
                    accelerator.is_main_process
                    and args.save_steps
                    #and global_step
                    and global_step % args.save_steps == 0
            ):
                checkpointandsave()
                # this will mess up the pbar if done here instead of end of epoch

            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optim.step(); lr_sched.step(); optim.zero_grad()

            global_step += 1
            if global_step >= args.max_steps:
                break

        pbar.close()
        if global_step >= args.max_steps:
            break

    if accelerator.is_main_process:
        if tb_writer is not None:
            tb_writer.close()
        pipe.text_encoder = None
        pipe.save_pretrained(args.output_dir, safe_serialization=True)
        print(f"finished:model saved to {args.output_dir}")

if __name__ == "__main__":
    main()

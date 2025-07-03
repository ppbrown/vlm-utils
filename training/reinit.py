# reinit_qk.py
#
# Re-initialise the UNet *cross-attention* query/key projections only.
# keeps all convolutions, value/out projections, etc., intact
# works with SD 1.5 UNet loaded through diffusers >=0.26
#
# Usage:
#   from diffusers import DiffusionPipeline
#   pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.bfloat16)
#   reinit_qk(pipe.unet)          # hard reset (Xavier)    
#   reinit_qk(pipe.unet, method="noise", sigma=0.1)  # soft reset (Gaussian noise)

import torch
from torch import nn
import torch.nn.init as init
from diffusers.models.attention import Attention as CrossAttention

def _xavier(m: nn.Linear) -> None:
    nn.init.xavier_uniform_(m.weight)
    if m.bias is not None:
        nn.init.zeros_(m.bias)

def _add_noise(m: nn.Linear, sigma: float) -> None:
    with torch.no_grad():
        std = m.weight.std().item()
        m.weight.add_(torch.randn_like(m.weight) * sigma * std)
        if m.bias is not None:
            m.bias.add_(torch.randn_like(m.bias) * sigma * std)



# randomize ALL unet weights not just qk
def reinit_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        init.normal_(m.weight, mean=0.0, std=0.02)


def reinit_qk(unet, *, method: str = "xavier", sigma: float = 0.1) -> None:
    """
    Reset (or perturb) the to_q / to_k projections in every CrossAttention
    layer that is actually used for **text-conditioning**.

    Args:
        unet   - diffusers.models.UNet2DConditionModel
        method - "xavier" (hard reset) or "noise" (additive Gaussian)
        sigma  - stdev factor for the noise method (default 0.1)

    Returns: None (UNet modified in-place)
    """

    if method.lower() == "all":
        print("ERROR: should not be calling reinit_qk with 'all' mode")
        exit(1)
        return

    hard = method.lower() == "xavier"
    affected = 0

    for mod in unet.modules():
        if isinstance(mod, CrossAttention) and getattr(mod, "is_cross_attention", False):
            for proj_name in ("to_q", "to_k"):
                proj: nn.Linear | None = getattr(mod, proj_name, None)
                if proj is None:
                    continue
                if hard:
                    _xavier(proj)
                else:
                    _add_noise(proj, sigma)
                affected += proj.weight.numel()

    print(f"[reinit_qk] updated {affected/1e6:.2f} M params in query/key projections")


def reinit_cross_attention(unet, *, method: str = "xavier", sigma: float = 0.1) -> None:
    """
    Reinitialize q, k, v, and output projections for all cross-attention modules in the UNet.
    """
    hard = method.lower() == "xavier"
    affected = 0

    for mod in unet.modules():
        if isinstance(mod, CrossAttention) and getattr(mod, "is_cross_attention", False):
            for proj_name in ("to_q", "to_k", "to_v"):
                proj: nn.Linear | None = getattr(mod, proj_name, None)
                if proj is None:
                    continue
                if hard:
                    _xavier(proj)
                else:
                    _add_noise(proj, sigma)
                affected += proj.weight.numel()
            # Output projection
            proj = getattr(mod, "to_out", None)
            if proj is not None:
                # If Sequential, usually first is Linear
                if isinstance(proj, nn.Sequential) and isinstance(proj[0], nn.Linear):
                    if hard:
                        _xavier(proj[0])
                    else:
                        _add_noise(proj[0], sigma)
                    affected += proj[0].weight.numel()
                elif isinstance(proj, nn.Linear):
                    if hard:
                        _xavier(proj)
                    else:
                        _add_noise(proj, sigma)
                    affected += proj.weight.numel()

    print(f"[reinit_attention_all] updated {affected/1e6:.2f} M params in q/k/v/out projections")


# I think "cross attention" is specifically for text emb mapping.
# Whereas self attention is for interpreting latent noisy images
def reinit_all_attention(
    unet,
    method: str = "xavier",
    sigma: float = 0.1,
    cross: bool = True,
    self_attn: bool = True,
):
    """
    Reinitialize q, k, v, and output projections for all attention modules in the UNet.
    By default, both cross-attention and self-attention modules are reinitialized.
    Set `cross` or `self_attn` to False to skip that type.
    """
    hard = method.lower() == "xavier"
    affected = 0

    for mod in unet.modules():
        # Only target modules that have is_cross_attention attribute (typical in SD1.5 code)
        if hasattr(mod, "is_cross_attention"):
            # Decide if this attention should be reset
            if (mod.is_cross_attention and cross) or (not mod.is_cross_attention and self_attn):
                # q, k, v
                for proj_name in ("to_q", "to_k", "to_v"):
                    proj = getattr(mod, proj_name, None)
                    if proj is None:
                        continue
                    if hard:
                        torch.nn.init.xavier_uniform_(proj.weight)
                        if proj.bias is not None:
                            torch.nn.init.zeros_(proj.bias)
                    else:
                        proj.weight.data += torch.randn_like(proj.weight) * sigma
                        if proj.bias is not None:
                            proj.bias.data += torch.randn_like(proj.bias) * sigma
                    affected += proj.weight.numel()
                # Output projection
                proj = getattr(mod, "to_out", None)
                if proj is not None:
                    if isinstance(proj, torch.nn.Sequential) and isinstance(proj[0], torch.nn.Linear):
                        if hard:
                            torch.nn.init.xavier_uniform_(proj[0].weight)
                            if proj[0].bias is not None:
                                torch.nn.init.zeros_(proj[0].bias)
                        else:
                            proj[0].weight.data += torch.randn_like(proj[0].weight) * sigma
                            if proj[0].bias is not None:
                                proj[0].bias.data += torch.randn_like(proj[0].bias) * sigma
                        affected += proj[0].weight.numel()
                    elif isinstance(proj, torch.nn.Linear):
                        if hard:
                            torch.nn.init.xavier_uniform_(proj.weight)
                            if proj.bias is not None:
                                torch.nn.init.zeros_(proj.bias)
                        else:
                            proj.weight.data += torch.randn_like(proj.weight) * sigma
                            if proj.bias is not None:
                                proj.bias.data += torch.randn_like(proj.bias) * sigma
                        affected += proj.weight.numel()
    print(f"Reinitialized {affected} attention parameters ({'cross' if cross else ''}{' & ' if cross and self_attn else ''}{'self' if self_attn else ''}-attention).")

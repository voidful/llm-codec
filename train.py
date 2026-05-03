# -*- coding: utf-8 -*-
import argparse
import os
import time
from pathlib import Path
from typing import Optional, List

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch import autograd

from llm_codec.gan import MPD, MSD, gan_d_loss, gan_g_loss, hinge_d_loss, hinge_g_loss, feature_matching_loss, set_requires_grad
from llm_codec.losses import MelLoss, STFTMagLoss, ComplexSTFTLoss, MultiScaleMelLoss, MultiResolutionSTFTLoss
from llm_codec.schedules import (
    cosine_with_warmup, lambda_ramp, cosine_ramp, cosine_decay
)
from llm_codec.system import AUVSystem
from llm_codec.utils import (
    ensure_dir, set_seed, save_wav,
    count_unique_1d, entropy_1d, token_topk_table, mel_image
)
from llm_codec.wb import WB


# [REMOVED] AR Curriculum related (ar_curriculum_k, subsample_codes)

def align_feat_to_T(x: torch.Tensor, T: int) -> torch.Tensor:
    if x.size(-1) == T:
        return x
    return F.adaptive_avg_pool1d(x, T)


def per_sample_rms_norm(wav: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    rms = torch.sqrt((wav ** 2).mean(dim=-1, keepdim=True) + eps)
    return wav / (rms + eps)


def d_r1_penalty(d_out_real_list: List[torch.Tensor], real_audio: torch.Tensor) -> torch.Tensor:
    if not isinstance(d_out_real_list, list):
        d_out_real_list = [d_out_real_list]
    total = sum([o.float().sum() for o in d_out_real_list])
    (grad_real,) = autograd.grad(
        outputs=total, inputs=real_audio, create_graph=True, retain_graph=True, only_inputs=True
    )
    r1 = grad_real.pow(2).reshape(grad_real.size(0), -1).sum(1).mean()
    return r1


def random_phase_roll(wav: torch.Tensor, max_shift: int) -> torch.Tensor:
    if max_shift <= 0:
        return wav
    B = wav.size(0)
    shifts = torch.randint(-max_shift, max_shift + 1, (B,), device=wav.device)
    out = []
    for i in range(B):
        out.append(torch.roll(wav[i:i + 1], int(shifts[i].item()), dims=-1))
    return torch.cat(out, dim=0)


@torch.no_grad()
def noisify_codes(codes: torch.Tensor, n_vocab: int, p_rand: float = 0.015, p_swap: float = 0.0) -> torch.Tensor:
    B, T = codes.shape
    out = codes.clone()
    if p_rand > 0:
        m = torch.rand(B, T, device=codes.device) < p_rand
        rnd = torch.randint(0, n_vocab, (B, T), device=codes.device)
        out[m] = rnd[m]
    if p_swap > 0 and T >= 2:
        m2 = torch.rand(B, T - 1, device=codes.device) < p_swap
        a = out[:, :-1].clone()
        b = out[:, 1:].clone()
        a[m2], b[m2] = b[m2], a[m2]
        out[:, :-1] = a
        out[:, 1:] = b
    return out


# ---------- Safety Helpers ----------
def sanitize_audio(x: torch.Tensor, clip: float):
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if clip > 0:
        x = torch.clamp(x, min=-clip, max=clip)
    return x


def match_length(x: torch.Tensor, T: int) -> torch.Tensor:
    """
    Align the last dimension of x to length T.
    If x is longer, crop it. If shorter, pad with reflection.
    This ensures recon matches input wav length exactly.
    """
    cur = x.size(-1)
    if cur == T:
        return x
    if cur > T:
        return x[..., :T]
    pad = T - cur
    return F.pad(x, (0, pad), mode="reflect")


def safe_softmax(logits: torch.Tensor, temp: float, eps: float = 1e-8):
    z = (logits / max(temp, eps))
    z = torch.clamp(z, min=-80.0, max=80.0)
    p = torch.softmax(z, dim=-1)
    p = torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    s = p.sum(dim=-1, keepdim=True).clamp_min(eps)
    return p / s


def finite_or_zero(x, name: str, step: int, wb: Optional[WB], enable: bool):
    # Handle case where x is a Python float instead of a tensor
    if not isinstance(x, torch.Tensor):
        # If it's a Python float/int, it's always finite (unless inf/nan)
        import math
        if math.isfinite(x):
            return x, False
        if enable and wb is not None:
            wb.log({f"nan/{name}": 1.0}, step=step, commit=False)
        return 0.0, True
    # Original tensor handling
    if torch.isfinite(x).all():
        return x, False
    if enable and wb is not None:
        wb.log({f"nan/{name}": 1.0}, step=step, commit=False)
    return x.new_tensor(0.0), True


def safe_detach(x):
    """Safely detach a value - handles both tensors and Python floats."""
    if isinstance(x, torch.Tensor):
        return x.detach()
    return x


# ---------- Safe Token Extraction (avoid in-place version error) ----------
@torch.no_grad()
def take_codes_safe(tokens: torch.Tensor, vocab: int) -> torch.Tensor:
    """
    1) detach() from graph
    2) clamp (non in-place)
    3) contiguous() ensures own storage
    """
    with torch.no_grad():
        c = tokens.detach().to(dtype=torch.long)
        c = torch.clamp(c, 0, vocab - 1).contiguous()
    return c


# ---------- AR logits -> Audio Vocab Cropping ----------
AUDIO_ID_TABLE = None


@torch.no_grad()
def build_audio_id_table(system, n_audio_tokens: int, device: torch.device) -> torch.Tensor:
    codes = torch.arange(n_audio_tokens, device=device).view(1, -1)
    ids = system.ar.codes_to_ids(codes).view(-1).long()
    return ids


def select_audio_logits(full_logits: torch.Tensor, id_table: torch.Tensor) -> torch.Tensor:
    return full_logits.index_select(dim=-1, index=id_table.to(full_logits.device))


def ensure_BxTxV(x: torch.Tensor, vocab: int) -> torch.Tensor:
    global AUDIO_ID_TABLE
    if x.dim() != 3:
        raise RuntimeError(f"expect 3D logits (B,T,V), got shape={tuple(x.shape)}")
    if x.size(-1) == vocab:
        return x
    if (AUDIO_ID_TABLE is not None) and (AUDIO_ID_TABLE.numel() == vocab):
        return select_audio_logits(x, AUDIO_ID_TABLE)
    raise RuntimeError(
        f"AR logits has incompatible shape={tuple(x.shape)}; none of dims equals vocab={vocab}"
    )



# =================================================
# Gumbel-Softmax Bridge (supports: gumbel / ste / detach)
# =================================================
class GumbelBridge(nn.Module):
    def __init__(self, codec_dim: int, n_audio_tokens: int, mode: str = "gumbel"):
        super().__init__()
        self.proj = nn.Linear(codec_dim, n_audio_tokens)
        assert mode in ("gumbel", "ste", "detach"), f"Unknown bridge_mode: {mode}"
        self.mode = mode

    def forward(self, quantized: torch.Tensor, embedding_weight: torch.Tensor, temp: float = 1.0, hard: bool = True):
        # quantized: (B, T, C)
        # Ensure input matches layer dtype
        dtype = self.proj.weight.dtype
        q = quantized.to(dtype)
        logits = self.proj(q) # (B, T, V)

        if self.mode == "gumbel":
            # Gumbel-Softmax (Hard=True for discrete forward, soft backward)
            y = F.gumbel_softmax(logits, tau=temp, hard=hard, dim=-1)
        elif self.mode == "ste":
            # Straight-Through Estimator: argmax forward, softmax gradient backward
            soft = F.softmax(logits / max(temp, 1e-8), dim=-1)
            idx = soft.argmax(dim=-1)  # (B, T)
            y_hard = F.one_hot(idx, logits.size(-1)).to(soft.dtype)  # (B, T, V)
            y = y_hard - soft.detach() + soft   # STE trick
        else:  # detach
            # Argmax with full detach: no gradient flows back to codec encoder
            with torch.no_grad():
                idx = logits.argmax(dim=-1)  # (B, T)
                y = F.one_hot(idx, logits.size(-1)).float()  # (B, T, V)

        # Get embeddings: (B, T, V) @ (V, H) -> (B, T, H)
        y_casted = y.to(embedding_weight.dtype)
        inputs_embeds = torch.matmul(y_casted, embedding_weight)
        
        return inputs_embeds, logits, y

# =================================================
# Medusa Head for Multi-Token Prediction
# =================================================
class MedusaHead(nn.Module):
    """
    Medusa-style prediction head for future token prediction.
    Initialized from original lm_head weights for faster convergence.
    """
    def __init__(self, hidden_size: int, vocab_size: int, init_weights: torch.Tensor = None):
        super().__init__()
        self.proj = nn.Linear(hidden_size, vocab_size, bias=False)
        if init_weights is not None:
            with torch.no_grad():
                self.proj.weight.data.copy_(init_weights)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Ensure input matches head dtype to avoid BF16/FP32 mismatch crashes
        target_dtype = self.proj.weight.dtype
        if hidden_states.dtype != target_dtype:
            hidden_states = hidden_states.to(target_dtype)
        return self.proj(hidden_states)

# =================================================
# Helper Functions
# =================================================

def get_required_audio_tokens(dataset, max_samples: int = 1000) -> int:
    """
    Auto-detect the maximum audio token ID in the dataset.
    Handles both List[int] and List[List[int]] formats for audio_codes.
    """
    max_token = 0
    count = 0
    
    for example in dataset:
        if count >= max_samples:
            break
        
        codes = example.get("audio_codes")
        if codes is None:
            continue
        
        # Check if codes is List[List[int]] or List[int]
        if isinstance(codes, list) and len(codes) > 0:
            # Check if first element is iterable (List[List[int]])
            if isinstance(codes[0], (list, tuple)):
                # List[List[int]] format
                for part in codes:
                    if len(part) > 0:
                        max_token = max(max_token, max(part))
            else:
                # List[int] format
                max_token = max(max_token, max(codes))
        
        count += 1
    
    return max_token


def main():
    p = argparse.ArgumentParser()
    # Data
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--seconds", type=float, default=4.0)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=8)

    # AUV & AR
    p.add_argument("--auv_ckpt", type=str, required=True)
    p.add_argument("--qwen_model", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--ar_token_prefix", type=str, default="<CODEC_")
    p.add_argument("--n_audio_tokens", type=int, default=20480)

    # tokenizer / embeddings
    p.add_argument("--tok_dir", type=str, default="")
    p.add_argument("--mean_resizing", action="store_true")

    # Reconstruction
    p.add_argument("--lambda_mel", type=float, default=1.0)
    p.add_argument("--lambda_vq", type=float, default=1.0)

    # Multi-Scale Mel Loss
    p.add_argument("--enable_ms_mel", action="store_true")
    p.add_argument("--lambda_ms_mel", type=float, default=1.0)

    # Multi-Resolution STFT Loss
    p.add_argument("--enable_mr_stft", action="store_true")
    p.add_argument("--lambda_mr_stft", type=float, default=1.0)

    # Complex STFT
    p.add_argument("--enable_cstft", action="store_true")
    p.add_argument("--lambda_cstft", type=float, default=0.0)
    p.add_argument("--cstft_phase_weight", type=float, default=0.5)
    p.add_argument("--mel_rms_norm", action="store_true")

    # LLM (embedding FT)
    p.add_argument("--lambda_ftp", type=float, default=1.0)

    p.add_argument("--ftp_ramp", type=int, default=2000,
                    help="FTP ramp-up duration in steps after ftp_delay_steps (default=2000).")
    p.add_argument("--ftp_delay_steps", type=int, default=5000)
    p.add_argument("--ftp_k", type=int, default=5,
                   help="FTP (Future Token Prediction) lookahead K (default=5). Weight decays for farther tokens.")

    # SA (Semantic Alignment)
    p.add_argument("--lambda_sa_cosine", type=float, default=0.1,
                   help="Weight for semantic cosine similarity loss (audio-text alignment)")
    p.add_argument("--lambda_sa_contrast", type=float, default=0.05,
                   help="Weight for semantic contrastive loss (discriminability)")
    p.add_argument("--sa_logit_scale", type=float, default=5.0,
                   help="Logit scale (1/temperature) for contrastive loss (default=5.0, was 14.28)")
    p.add_argument("--sa_label_smoothing", type=float, default=0.1,
                   help="Label smoothing for contrastive cross-entropy loss")
    p.add_argument("--sa_ema_momentum", type=float, default=0.99,
                   help="EMA momentum for memory bank updates (0=no EMA, use FIFO)")
    p.add_argument("--sa_delay_steps", type=int, default=5000,
                   help="Delay before semantic loss kicks in")
    p.add_argument("--sa_warmup", type=int, default=5000,
                   help="Warmup steps for semantic loss ramp-up")
    p.add_argument("--sa_queue_size", type=int, default=512,
                   help="Size of memory bank queue for contrastive learning (default=512)")
    p.add_argument("--bridge_mode", type=str, choices=["gumbel", "ste", "detach"], default="gumbel",
                   help="Bridge mode: gumbel (default), ste (straight-through estimator), detach (no gradient to codec)")
    p.add_argument("--gumbel_temp_init", type=float, default=1.0)
    p.add_argument("--gumbel_temp_final", type=float, default=0.3)
    p.add_argument("--gumbel_temp_steps", type=int, default=20000)

    # Reconstruction Schedule
    p.add_argument("--mel_decay_final", type=float, default=0.5,
                   help="Final lambda_mel after decay (relative to initial)")
    p.add_argument("--mel_decay_steps", type=int, default=50000,
                   help="Steps for mel loss decay schedule (0=no decay)")

    # GAN
    p.add_argument("--enable_gan", action="store_true")
    p.add_argument("--lambda_gan", type=float, default=0.0)
    p.add_argument("--d_only_steps", type=int, default=0)
    p.add_argument("--gan_warmup", type=int, default=4000)
    p.add_argument("--gan_ramp", type=int, default=16000)
    p.add_argument("--lambda_fm_init", type=float, default=3.0)
    p.add_argument("--lambda_fm_final", type=float, default=1.0)

    p.add_argument("--lr_d", type=float, default=1e-4)
    p.add_argument("--gan_loss_type", type=str, choices=["lsgan", "hinge"], default="hinge")
    p.add_argument("--d_update_every", type=int, default=2)
    p.add_argument("--r1_gamma", type=float, default=10.0)
    p.add_argument("--d_reg_every", type=int, default=16)

    # Phase Jitter
    p.add_argument("--enable_phase_jitter", action="store_true")
    p.add_argument("--phase_jitter_max", type=int, default=0)

    # Opt
    p.add_argument("--lr_enc", type=float, default=1e-4)
    p.add_argument("--lr_dec", type=float, default=5e-5)
    p.add_argument("--lr_embed", type=float, default=5e-4)
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--lr_total_steps", type=int, default=200000)
    p.add_argument("--lr_warmup", type=int, default=2000)

    # AMP / device
    p.add_argument("--mp_dtype", type=str, choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=1337)

    # Loop
    p.add_argument("--max_steps", type=int, default=200000)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--val_every", type=int, default=2000)
    p.add_argument("--save_every", type=int, default=10000)

    # GAN Safety Pause
    p.add_argument("--fm_pause_share", type=float, default=0.35)
    p.add_argument("--gan_pause_steps", type=int, default=500)

    # I/O
    p.add_argument("--out_dir", type=str, default="runs/auv_qwen_ar")
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--dump_each_ckpt", action="store_true")

    # WandB
    p.add_argument("--wandb_project", type=str, default="codec-ar")
    p.add_argument("--wandb_run", type=str, default="train")
    p.add_argument("--wandb_offline", action="store_true")

    # Others: Grad Accum & NaN Guard & GAN Precision
    p.add_argument("--grad_accum_steps", type=int, default=8)
    p.add_argument("--nan_guard", action="store_true")
    p.add_argument("--clip_recon_amp", type=float, default=1.2)
    p.add_argument("--gan_amp", type=str, choices=["auto", "fp32", "amp"], default="amp")
    p.add_argument("--opt_codec", type=str, default="adamw", choices=["adamw", "sgd"], help="Optimizer for Codec")

    args = p.parse_args()

    # CPU threads
    try:
        import psutil
        cpu_physical = psutil.cpu_count(logical=False) or os.cpu_count() or 8
    except Exception:
        cpu_physical = os.cpu_count() or 8
    torch.set_num_threads(max(1, min(4, cpu_physical // 2)))
    os.environ.setdefault("OMP_NUM_THREADS", str(torch.get_num_threads()))

    ensure_dir(args.out_dir)
    set_seed(args.seed)
    device = torch.device(args.device)

    tok_dir = args.tok_dir if args.tok_dir else os.path.join(args.out_dir, "tokenizer")
    ensure_dir(tok_dir)

    accum_steps = max(1, int(args.grad_accum_steps))

    # AMP
    if args.mp_dtype == "bf16":
        amp_dtype = torch.bfloat16
        use_scaler = False
        print("[AMP] BF16.")
    elif args.mp_dtype == "fp16":
        amp_dtype = torch.float16
        use_scaler = True
        print("[AMP] FP16.")
    else:
        amp_dtype = torch.float32
        use_scaler = False
        print("[AMP] FP32.")

    # System
    system = AUVSystem(
        ckpt_path=args.auv_ckpt, device=device, bf16=(args.mp_dtype == "bf16"),
        qwen_model=args.qwen_model, mp_dtype=args.mp_dtype,
        n_audio_tokens=args.n_audio_tokens, ar_token_prefix=args.ar_token_prefix,
        tok_dir=tok_dir, mean_resizing=args.mean_resizing
    ).to(device)

    # Fix BatchNorm to eval checks
    def set_batchnorm_to_eval(m):
        classname = m.__class__.__name__
        if "BatchNorm" in classname:
            m.eval()

    system.codec.model.apply(set_batchnorm_to_eval)
    print("[BN Fix] Set all BatchNorm layers in system.codec.model to eval mode.")

    # Sample Rate / Hop Alignment (do not modify system attributes)
    codec_module = getattr(system, "codec", None)
    codec_model = getattr(codec_module, "model", codec_module)

    codec_sr = int(getattr(codec_module, "sample_rate",
                           getattr(codec_model, "sample_rate", getattr(system, "sample_rate", 16000))))
    codec_hop = int(getattr(codec_module, "hop_length",
                            getattr(codec_model, "hop_length", getattr(system, "hop_length", 320))))

    target_sr = codec_sr
    target_hop = codec_hop

    print(f"[SR] Using target_sr={target_sr}, target_hop={target_hop} as sample_rate/hop_length for data and recon")

    # Audio code -> Global vocab id table
    global AUDIO_ID_TABLE
    AUDIO_ID_TABLE = build_audio_id_table(system, args.n_audio_tokens, device)
    print(f"[AR] audio vocab id table built: len={AUDIO_ID_TABLE.numel()} (head={AUDIO_ID_TABLE[:5].tolist()})")

    if target_hop <= 0:
        print("[align][warn] hop_length <= 0, please check tokenizer/codec.")
    else:
        print(f"[align] codec hop_length = {target_hop}")

    # GAN
    mpd = None
    msd = None
    opt_d = None
    if args.enable_gan:
        mpd = MPD().to(device).float()
        msd = MSD().to(device).float()
        opt_d = torch.optim.AdamW(list(mpd.parameters()) + list(msd.parameters()),
                                  lr=args.lr_d, betas=(0.8, 0.99), weight_decay=0.0)

    # Losses
    mel_loss = MelLoss(sr=target_sr, n_fft=1024, hop_length=target_hop, n_mels=100).to(device)
    ms_mel_loss = MultiScaleMelLoss(sr=target_sr).to(device) if args.enable_ms_mel else None
    mr_stft_loss = MultiResolutionSTFTLoss().to(device) if args.enable_mr_stft else None
    cstft_loss = ComplexSTFTLoss(phase_weight=args.cstft_phase_weight).to(device) if args.enable_cstft else None

    # Gumbel Bridge
    # We need codec_dim. Let's run a dummy encode to get it.
    with torch.no_grad():
        dummy_wav = torch.randn(1, 1, 16000).to(device)
        dummy_enc = system.codec.encode(dummy_wav, sr=16000)
        # quantized has shape (B, T, C) where C is the codec dimension
        codec_dim = dummy_enc["quantized"].shape[-1]
        print(f"[Gumbel] Detected Codec Dim: {codec_dim} (from shape {dummy_enc['quantized'].shape})")

    n_audio_tokens = args.n_audio_tokens
    gumbel_bridge = GumbelBridge(codec_dim, n_audio_tokens, mode=args.bridge_mode).to(device)
    print(f"[Bridge] mode={args.bridge_mode}, {codec_dim} -> {n_audio_tokens}")

    # ==============================================
    # Medusa Heads for Multi-Token Prediction
    # ==============================================
    hidden_size = system.ar.model.config.hidden_size
    # Get audio token IDs for extracting lm_head weights
    audio_ids_for_init = system.ar.audio_token_id_table  # (N_audio,)
    # Extract lm_head weights for audio tokens only
    lm_head_weight_audio = system.ar.model.lm_head.weight.data[audio_ids_for_init].clone()  # (N_audio, H)
    
    # Get the dtype of lm_head to ensure consistency
    lm_head_dtype = system.ar.model.lm_head.weight.dtype
    
    medusa_heads = nn.ModuleList([
        MedusaHead(hidden_size, n_audio_tokens, init_weights=lm_head_weight_audio)
        for _ in range(args.ftp_k)
    ]).to(device=device, dtype=lm_head_dtype)  # Match lm_head dtype (e.g., BFloat16)
    print(f"[FTP] Initialized {args.ftp_k} Medusa heads with lm_head weights, hidden_size={hidden_size}, dtype={lm_head_dtype}")


    # Opt
    if not (hasattr(system.codec.model, "tokenizer") and hasattr(system.codec.model, "token2wav")):
        raise RuntimeError("Model structure changed, cannot find 'tokenizer' or 'token2wav'. Please check system.codec.model.")

    params_enc = [p for p in system.codec.model.tokenizer.parameters() if p.requires_grad]
    # Add Gumbel Bridge params to Encoder Optimizer
    params_enc += list(gumbel_bridge.parameters())
    params_dec = [p for p in system.codec.model.token2wav.parameters() if p.requires_grad]

    all_codec_params = set(p for p in system.codec.model.parameters() if p.requires_grad)
    opt_params = set(params_enc) | set(params_dec)
    missing_params = all_codec_params - opt_params
    if missing_params:
        print(f"[warn] {len(missing_params)} codec params not included in 'opt_enc' or 'opt_dec'.")

    print(f"[Opt] Encoder ('tokenizer') params: {len(params_enc)}")
    print(f"[Opt] Decoder ('token2wav') params: {len(params_dec)}")

    if getattr(args, "opt_codec", "adamw") == "sgd":
        print("[Opt] Using pure SGD (+Momentum 0.9) for Encoder and Decoder to strictly scale steps by gradient magnitude.")
        opt_enc = torch.optim.SGD(params_enc, lr=args.lr_enc, momentum=0.9, weight_decay=1e-4)
        opt_dec = torch.optim.SGD(params_dec, lr=args.lr_dec, momentum=0.9, weight_decay=1e-4)
    else:
        opt_enc = torch.optim.AdamW(params_enc, lr=args.lr_enc, betas=(0.9, 0.99), weight_decay=1e-4)
        opt_dec = torch.optim.AdamW(params_dec, lr=args.lr_dec, betas=(0.9, 0.99), weight_decay=1e-4)

    # Include Medusa heads in opt_embed
    ar_params = list(system.ar.model.parameters())
    medusa_params = list(medusa_heads.parameters())
    opt_embed = torch.optim.AdamW(
        [p for p in ar_params if p.requires_grad] + medusa_params,
        lr=args.lr_embed, betas=(0.9, 0.99), weight_decay=0.0
    )
    print(f"[Opt] Medusa heads params: {sum(p.numel() for p in medusa_params)}")
    scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)


    # WB
    wb_enabled = (args.wandb_project != "")
    run_name = args.wandb_run if args.wandb_run else Path(args.out_dir).name
    if args.wandb_offline:
        os.environ["WANDB_MODE"] = "offline"
    wb = WB(enabled=wb_enabled, project=args.wandb_project or "debug", run_name=run_name, dir_=args.out_dir,
            config=vars(args))
    wb.watch(system.codec.model.tokenizer, log="parameters", log_freq=500)
    wb.watch(system.codec.model.token2wav, log="parameters", log_freq=500)
    wb.watch(system.ar.model.get_input_embeddings(), log="parameters", log_freq=500)
    if mpd is not None:
        wb.watch(mpd, log="parameters", log_freq=500)
    if msd is not None:
        wb.watch(msd, log="parameters", log_freq=500)

    # Data
    def _collate_on_the_fly(examples, target_sr: int, seconds: float, phase_jitter: int = 0):
        import torch as _torch
        import torchaudio
        import torch.nn.functional as _F
        T = int(seconds * target_sr)
        wavs = []
        texts = []
        for ex in examples:
            a = ex["audio"]
            t = ex.get("text", "")
            if not t and "normalized_text" in ex:
                t = ex["normalized_text"]
            texts.append(t)

            wav = _torch.tensor(a["array"], dtype=_torch.float32)
            if wav.ndim == 1:
                wav = wav.unsqueeze(0)
            if wav.size(0) > 1:
                wav = wav.mean(0, keepdim=True)
            sr = int(a["sampling_rate"])
            if sr != target_sr:
                wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
            if wav.shape[1] < T:
                pad = T - wav.shape[1]
                left = pad // 2
                right = pad - left
                wav = _F.pad(wav, (left, right), mode="reflect")
            elif wav.shape[1] > T:
                s = _torch.randint(0, wav.shape[1] - T + 1, (1,)).item()
                wav = wav[:, s:s + T]
            if phase_jitter > 0:
                shift = int(_torch.randint(-phase_jitter, phase_jitter + 1, (1,)).item())
                if shift != 0:
                    wav = _torch.roll(wav, shifts=shift, dims=-1)
            wavs.append(wav)
        return {"wav": _torch.stack(wavs, dim=0), "text": texts}

    ds_train = load_dataset("librispeech_asr", "clean", split="train.100", cache_dir=args.cache_dir)
    ds_val = load_dataset("librispeech_asr", "clean", split="validation", cache_dir=args.cache_dir)
    pf = 4 if args.num_workers > 0 else None

    phase_jitter_for_loader = args.phase_jitter_max if args.enable_phase_jitter else 0

    tr_loader = torch.utils.data.DataLoader(
        ds_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=True, persistent_workers=(args.num_workers > 0),
        drop_last=True,
        collate_fn=lambda ex: _collate_on_the_fly(ex, target_sr, args.seconds,
                                                  phase_jitter=phase_jitter_for_loader),
        prefetch_factor=pf)
    va_loader = torch.utils.data.DataLoader(
        ds_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=True, persistent_workers=(args.num_workers > 0),
        collate_fn=lambda ex: _collate_on_the_fly(ex, target_sr, args.seconds, phase_jitter=0),
        prefetch_factor=pf)

    # Resume
    start_step = 0
    if args.resume and Path(args.resume).is_file():
        ckpt = torch.load(args.resume, map_location='cpu')
        system.codec.model.load_state_dict(ckpt["auv"])
        emb = ckpt["qwen_embed"]
        cur_V = system.ar.model.get_input_embeddings().weight.shape[0]
        if emb.shape[0] != cur_V:
            raise RuntimeError(f"[resume] tokenizer vocab_size ({cur_V}) != saved embed ({emb.shape[0]}).")
        system.ar.model.get_input_embeddings().weight.data.copy_(emb)
        if "opt_enc" in ckpt and "opt_dec" in ckpt:
            opt_enc.load_state_dict(ckpt["opt_enc"])
            opt_dec.load_state_dict(ckpt["opt_dec"])
        elif "opt_auv" in ckpt:
            print("[resume][warn] Resuming from old 'opt_auv', might be inaccurate.")
            opt_enc.load_state_dict(ckpt["opt_auv"])
            opt_dec.load_state_dict(ckpt["opt_auv"])
        opt_embed.load_state_dict(ckpt["opt_embed"])

        if args.enable_gan and "mpd" in ckpt and "msd" in ckpt and "opt_d" in ckpt:
            try:
                mpd.load_state_dict(ckpt["mpd"])
                msd.load_state_dict(ckpt["msd"])
                opt_d.load_state_dict(ckpt["opt_d"])
            except Exception as e:
                print(f"[resume][warn] Discriminator resume failed (ignored): {e}")
        start_step = ckpt.get("step", 0)
        print(f"[resume] from {args.resume} @ step {start_step}")

    # Reference Samples
    with torch.no_grad():
        ex0_batch = next(iter(tr_loader))
        ex0 = ex0_batch["wav"][:1].to(device)

    def log_reference_precommit(step: int):
        with torch.no_grad():
            out = system(ex0, sr=target_sr)
            recon_ref = sanitize_audio(out.recon, clip=args.clip_recon_amp)
            recon_ref = match_length(recon_ref, ex0.size(-1))
        a0 = wb.audio(ex0, target_sr, caption=f"orig@{step}")
        a1 = wb.audio(recon_ref, target_sr, caption=f"codec_recon@{step}")
        logs = {"audio/orig": a0, "audio/codec_recon": a1}
        logs["mel/orig"] = wb.image(mel_image(ex0[0], target_sr, n_fft=1024, hop=target_hop),
                                    caption=f"mel-orig@{step}")
        logs["mel/recon"] = wb.image(mel_image(recon_ref[0], target_sr, n_fft=1024, hop=target_hop),
                                     caption=f"mel-recon@{step}")
        wb.log(logs, step=step, commit=False)

    # Train loop
    step = start_step
    
    if step < args.d_only_steps:
        print("[Init] Entering D-only warmup: Codec stays in train() mode (VQ EMA continues tracking), only skipping opt_enc/opt_dec.step().")

    tr_iter = iter(tr_loader)
    t0 = time.time()
    gan_pause_until = -1
    bad_micro_streak = 0

    # SA (Semantic Alignment) Memory Bank (FIFO Queue for Contrastive Learning)
    sa_queue_size = args.sa_queue_size  # Number of negative samples to compare against
    sa_queue_text = []  # Store past Text Embeddings

    def temp_schedule(step, t0=1.0, t1=0.3, s=20000):
        if step >= s:
            return t1
        u = step / float(s)
        return t1 + 0.5 * (t0 - t1) * (1 + math.cos(math.pi * u))

    # Initialize model to train mode
    system.train()
    # But lock Batch Norm if using codec (often needed for pretrained encoder/decoder)
    system.codec.model.apply(set_batchnorm_to_eval)

    while step < args.max_steps:
        opt_enc.zero_grad(set_to_none=True)
        opt_dec.zero_grad(set_to_none=True)
        opt_embed.zero_grad(set_to_none=True)

        lr_enc = cosine_with_warmup(step, args.lr_enc, args.lr_warmup, args.lr_total_steps)
        lr_dec = cosine_with_warmup(step, args.lr_dec, args.lr_warmup, args.lr_total_steps)
        lr_embed = cosine_with_warmup(step, args.lr_embed, args.lr_warmup, args.lr_total_steps)

        for g in opt_enc.param_groups:
            g["lr"] = lr_enc
        for g in opt_dec.param_groups:
            g["lr"] = lr_dec
        for g in opt_embed.param_groups:
            g["lr"] = lr_embed

        if opt_d is not None:
            for g in opt_d.param_groups:
                g["lr"] = args.lr_d

        last_vals = {}
        bad_micro = False
        wav_last = None
        recon_last = None

        for _ in range(accum_steps):
            try:
                batch = next(tr_iter)
            except StopIteration:
                tr_iter = iter(tr_loader)
                batch = next(tr_iter)

            wav = sanitize_audio(batch["wav"].to(device), clip=args.clip_recon_amp)
            texts = batch["text"]

            # Main Branch (Codec)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(amp_dtype != torch.float32)):
                out_main = system(wav, sr=target_sr)
                recon_clean = sanitize_audio(out_main.recon, clip=args.clip_recon_amp)
                recon_clean = match_length(recon_clean, wav.size(-1))
                wav_mel_src = per_sample_rms_norm(wav) if args.mel_rms_norm else wav
                recon_mel_src = per_sample_rms_norm(recon_clean) if args.mel_rms_norm else recon_clean

            with torch.autocast(device_type=device.type, enabled=False):
                L_mel_raw = mel_loss(wav_mel_src.float(), recon_mel_src.float())

            # Reconstruction schedule: gradually decrease mel weight to emphasize semantic later
            if args.mel_decay_steps > 0:
                lambda_mel_eff = cosine_decay(step, args.lambda_mel, 
                                              args.lambda_mel * args.mel_decay_final, 
                                              args.mel_decay_steps)
            else:
                lambda_mel_eff = args.lambda_mel
            L_mel = L_mel_raw * lambda_mel_eff
            L_ms_mel = recon_clean.new_tensor(0.0)
            if ms_mel_loss is not None and args.lambda_ms_mel > 0.0:
                with torch.autocast(device_type=device.type, enabled=False):
                    L_ms_mel_raw = ms_mel_loss(wav.float(), recon_clean.float())
                    L_ms_mel = L_ms_mel_raw * args.lambda_ms_mel


            # Multi-Resolution STFT Loss
            L_mr_stft = recon_clean.new_tensor(0.0)
            if mr_stft_loss is not None and args.lambda_mr_stft > 0.0:
                with torch.autocast(device_type=device.type, enabled=False):
                    L_mr_stft_raw = mr_stft_loss(wav.float(), recon_clean.float())
                    L_mr_stft = L_mr_stft_raw * args.lambda_mr_stft

            L_cstft = recon_clean.new_tensor(0.0)
            if cstft_loss is not None and args.lambda_cstft > 0.0:
                with torch.autocast(device_type=device.type, enabled=False):
                    L_cstft_raw = cstft_loss(wav.float(), recon_clean.float())
                    L_cstft = L_cstft_raw * args.lambda_cstft

            # VQ Loss
            vq_loss = out_main.vq_loss * args.lambda_vq

            # =====================================================================
            # Gumbel Bridge - UNIFIED for both LLM loss and Semantic loss
            # This must come BEFORE LLM loss so we can use inputs_embeds
            # =====================================================================
            
            # 1. Get embedding weight from AR model
            embed_weight = system.ar.model.get_input_embeddings().weight  # (V_total, H)
            
            # Construct a subset embedding matrix for audio tokens only
            # NOTE: audio_embed_weight must be OUTSIDE no_grad to allow gradients
            # to flow back to LLM audio token embeddings AND Codec (via quant_in)
            audio_ids_all = system.ar.audio_token_id_table  # (N_audio,) - just indices
            audio_embed_weight = embed_weight[audio_ids_all]  # (N_audio, H) - NEEDS gradients!
            
            # 2. Apply Gumbel Bridge to get differentiable embeddings
            # out_main.quant may be (B, C, Tf_high) or (B, Tf_high, C) depending on codec
            # out_main.tokens is (B, Tf_low) - lower resolution discrete tokens
            # We need to downsample quant to match token resolution
            quant_raw = out_main.quant  # Shape varies
            tokens_target = out_main.tokens  # (B, Tf_low)

            # [Fix] Reshape tokens if they are flattened (B=1) but quant implies B>1
            if tokens_target.dim() == 2 and tokens_target.shape[0] == 1 and quant_raw.shape[0] > 1:
                # Assume tokens are flattened as (B*T), reshape to (B, T)
                tokens_target = tokens_target.view(quant_raw.shape[0], -1)

            target_len = tokens_target.shape[1]
            
            # Detect tensor layout: codec_dim is 512, so whichever dim has 512 is C
            # Shape could be (B, C, T) where C=512 or (B, T, C) where C=512

            if quant_raw.shape[1] == 512:
                # Format is (B, C, T) - need to transpose to (B, T, C) for linear layer
                quant_T = quant_raw.shape[2]  # Time dimension is at index 2
                if quant_T != target_len:
                    # Downsample along time axis: (B, C, T_high) -> (B, C, T_low)
                    quant_downsampled = F.adaptive_avg_pool1d(quant_raw, target_len)
                else:
                    quant_downsampled = quant_raw
                # Transpose to (B, T, C) for linear layer
                quant_in = quant_downsampled.transpose(1, 2)
            else:
                # Format is (B, T, C) - time is at index 1
                quant_T = quant_raw.shape[1]
                if quant_T != target_len:
                    # Transpose to (B, C, T) for pooling, then back
                    quant_in = F.adaptive_avg_pool1d(
                        quant_raw.transpose(1, 2), target_len
                    ).transpose(1, 2)
                else:
                    quant_in = quant_raw
            
            # Temperature schedule for Gumbel-Softmax
            gumbel_temp = temp_schedule(step, args.gumbel_temp_init, args.gumbel_temp_final, args.gumbel_temp_steps)
            
            # inputs_embeds_gumbel: (B, Tf_low, H) - differentiable embeddings
            # logits_gumbel: (B, Tf_low, N_audio) - logits for auxiliary CE loss
            inputs_embeds_gumbel, logits_gumbel, y_gumbel = gumbel_bridge(
                quant_in, audio_embed_weight, temp=gumbel_temp, hard=True
            )
            
            # 3. Final length alignment (should be no-op after downsampling, but safety check)
            if inputs_embeds_gumbel.shape[1] != tokens_target.shape[1]:
                min_len = min(inputs_embeds_gumbel.shape[1], tokens_target.shape[1])
                inputs_embeds_gumbel = inputs_embeds_gumbel[:, :min_len, :]
                logits_gumbel = logits_gumbel[:, :min_len, :]
                tokens_target = tokens_target[:, :min_len]
            
            # 4. Auxiliary CE Loss: Gumbel logits should match VQ tokens
            loss_gumbel = F.cross_entropy(logits_gumbel.reshape(-1, n_audio_tokens), tokens_target.reshape(-1))
            
            # =====================================================================
            # Unified LLM Forward Pass (Optimized)
            # - Computes hidden_states once for both MTP and Semantic Loss
            # =====================================================================
            
            # SA warmup parameters
            sa_on = step >= args.sa_delay_steps
            lam_sa_cosine_eff = (lambda_ramp(step, args.sa_delay_steps, args.sa_warmup,
                                              args.lambda_sa_cosine) if sa_on else 0.0)
            lam_sa_contrast_eff = (lambda_ramp(step, args.sa_delay_steps, args.sa_warmup,
                                                args.lambda_sa_contrast) if sa_on else 0.0)
            ftp_on = int(step >= args.ftp_delay_steps) # Re-evaluate ftp_on here for clarity
            sa_enabled = (lam_sa_cosine_eff > 0 or lam_sa_contrast_eff > 0) and ftp_on

            # Initialize losses
            ftp_loss = wav.new_tensor(0.0)
            ftp_raw = wav.new_tensor(0.0)
            sa_cosine_loss = wav.new_tensor(0.0)
            sa_contrast_loss = wav.new_tensor(0.0)
            sa_cosine_raw = wav.new_tensor(0.0)
            sa_contrast_raw = wav.new_tensor(0.0)
            
            # Execute LLM forward if either MTP or Semantic Loss is active
            lam_ftp = (lambda_ramp(step, args.ftp_delay_steps, args.ftp_ramp,
                                   args.lambda_ftp) if ftp_on else 0.0)
            if lam_ftp > 0 or sa_enabled:
                try:
                    # Prepare inputs_embeds for autoregressive prediction
                    bos_id = system.ar.tok.bos_token_id
                    current_bs = inputs_embeds_gumbel.size(0)
                    seq_len = inputs_embeds_gumbel.size(1)
                    
                    # Add BOS token if present (Common for both MTP & Semantic)
                    if bos_id is not None:
                        bos_embed = system.ar.model.get_input_embeddings()(
                            torch.tensor([[bos_id]], device=device)
                        ).expand(current_bs, -1, -1)  # (B, 1, H)
                        llm_input_embeds = torch.cat([bos_embed, inputs_embeds_gumbel], dim=1)  # (B, 1+T, H)
                    else:
                        llm_input_embeds = inputs_embeds_gumbel
                    
                    # Single Forward Pass
                    llm_out = system.ar.model(
                        inputs_embeds=llm_input_embeds,
                        output_hidden_states=True,
                        use_cache=False
                    )
                    
                    # --- A. Future Token Prediction (FTP) ---
                    if lam_ftp > 0:
                        predict_k = args.ftp_k
                        raw_weights = [1.0 / (j + 1) for j in range(predict_k)]
                        weight_sum = sum(raw_weights)
                        k_weights = [w / weight_sum for w in raw_weights]

                        # Keep gradient flow for MTP - enables ftp_loss to update Codec and LLM embeddings
                        hidden_states_mtp = llm_out.hidden_states[-1]  # (B, 1+T, H)
                        
                        total_loss = wav.new_tensor(0.0)
                        valid_k = 0
                        
                        for j in range(predict_k):
                            max_pred_len = seq_len - j
                            if max_pred_len <= 0:
                                continue
                            
                            hidden_for_pred = hidden_states_mtp[:, :max_pred_len, :]
                            pred_logits = medusa_heads[j](hidden_for_pred)
                            target_labels = tokens_target[:, j:j + max_pred_len]
                            
                            loss_j = F.cross_entropy(
                                pred_logits.reshape(-1, pred_logits.size(-1)),
                                target_labels.reshape(-1)
                            )
                            total_loss = total_loss + k_weights[j] * loss_j
                            valid_k += 1
                        
                        ftp_raw = total_loss if valid_k > 0 else wav.new_tensor(0.0)
                        ftp_loss = ftp_raw * lam_ftp

                    # --- B. Semantic Alignment / SA (Middle Layers) ---
                    if sa_enabled:
                        # 1. Get Text Hidden States
                        valid_texts = [t if t and len(t.strip()) > 0 else " " for t in texts]
                        txt_enc = system.ar.tok(valid_texts, return_tensors="pt", padding=True, truncation=True).to(device)
                        
                        with torch.no_grad():
                            txt_out = system.ar.model(
                                input_ids=txt_enc.input_ids,
                                attention_mask=txt_enc.attention_mask,
                                output_hidden_states=True
                            )
                            txt_layers = txt_out.hidden_states[1:] 
                            last_token_indices = txt_enc.attention_mask.sum(dim=1) - 1

                        # 2. Get Audio Hidden States (From Unified Forward)
                        all_audio_layers = llm_out.hidden_states[1:] # L1 to L32
                        num_layers = len(all_audio_layers)
                        
                        # Align batch sizes (last batch of epoch may be smaller)
                        bs_text = txt_enc.input_ids.size(0)
                        bs_audio = all_audio_layers[0].size(0)
                        bs = min(current_bs, bs_text, bs_audio)
                        
                        # Semantic Loss on Middle-High Layers Only (L12 - L28)
                        sa_start = num_layers // 3
                        sa_end = int(num_layers * 0.8)
                        bank_layer_idx = sa_end - 1
                        
                        total_cosine_loss = wav.new_tensor(0.0)  # Must be tensor for gradient flow
                        total_contrast_loss = wav.new_tensor(0.0)  # Must be tensor for gradient flow
                        logit_scale = args.sa_logit_scale
                        
                        top_feat_text = None

                        for i in range(sa_start, sa_end):
                            layer_weight = 1.0 / (sa_end - sa_start)
                            
                            h_text = txt_layers[i]
                            feat_text = h_text[torch.arange(bs), last_token_indices[:bs]]
                            feat_audio = all_audio_layers[i][:bs, -1, :]

                            feat_text_norm = F.normalize(feat_text, p=2, dim=-1)
                            feat_audio_norm = F.normalize(feat_audio, p=2, dim=-1)

                            if i == bank_layer_idx:
                                # Mean-pool across batch → (H,) to avoid shape mismatch in queue
                                top_feat_text = feat_text_norm.detach().mean(dim=0)
                                top_feat_audio = feat_audio_norm.detach().mean(dim=0)

                            if lam_sa_cosine_eff > 0:
                                cos_sim = (feat_audio_norm * feat_text_norm).sum(dim=-1).mean()
                                loss_cosine = 1.0 - cos_sim
                                total_cosine_loss += loss_cosine * layer_weight

                            if lam_sa_contrast_eff > 0 and len(sa_queue_text) > 0:
                                neg_text_feats = torch.stack(sa_queue_text, dim=0).to(device)  # (Q, H)
                                pos_logit = (feat_audio_norm * feat_text_norm).sum(dim=-1, keepdim=True) * logit_scale  # (B, 1)
                                neg_logits = torch.matmul(feat_audio_norm, neg_text_feats.T) * logit_scale  # (B, Q)
                                all_logits = torch.cat([pos_logit, neg_logits], dim=1)  # (B, 1+Q)
                                contrast_labels = torch.zeros(bs, dtype=torch.long, device=device)
                                loss_contrast = F.cross_entropy(all_logits, contrast_labels, 
                                                               label_smoothing=args.sa_label_smoothing)
                                total_contrast_loss += loss_contrast * layer_weight

                        # Keep as tensors to preserve gradients
                        sa_cosine_raw = total_cosine_loss
                        sa_contrast_raw = total_contrast_loss
                        
                        # Scale by effective lambda (both are tensors now)
                        sa_cosine_loss = sa_cosine_raw * lam_sa_cosine_eff
                        sa_contrast_loss = sa_contrast_raw * lam_sa_contrast_eff

                        if top_feat_text is not None:
                            if args.sa_ema_momentum > 0 and len(sa_queue_text) > 0:
                                momentum = args.sa_ema_momentum
                                # Both are (H,) now — no batch dim mismatch possible
                                new_slot = (1 - momentum) * top_feat_text + momentum * sa_queue_text[-1]
                                sa_queue_text[-1] = new_slot.detach()
                                if len(sa_queue_text) < sa_queue_size:
                                    sa_queue_text.append(top_feat_text.detach())
                            else:
                                sa_queue_text.append(top_feat_text.detach())
                                if len(sa_queue_text) > sa_queue_size:
                                    sa_queue_text.pop(0)

                except Exception as e:
                    print(f"\n!!!! [LLM/Semantic Error] Step {step}: {e} !!!!\n")
                    # Reset losses on error to avoid crash
                    ftp_loss = wav.new_tensor(0.0)
                    sa_cosine_loss = wav.new_tensor(0.0)
                    sa_contrast_loss = wav.new_tensor(0.0)



            # GAN / FM
            lam_gan_eff = lambda_ramp(step, args.gan_warmup, args.gan_ramp, args.lambda_gan)
            if step < gan_pause_until:
                lam_gan_eff = 0.0
            lam_fm_eff = args.lambda_fm_init

            gan_g = recon_clean.new_tensor(0.0)
            fm = recon_clean.new_tensor(0.0)
            if args.enable_gan and (lam_gan_eff > 0.0 or lam_fm_eff > 0.0):
                if mpd is not None:
                    set_requires_grad(mpd, False)
                if msd is not None:
                    set_requires_grad(msd, False)
                with torch.autocast(device_type=device.type, enabled=False):
                    if mpd is not None:
                        mpd_out_fake, mpd_feat_fake = mpd(recon_clean.float())
                        mpd_out_real, mpd_feat_real = mpd(wav.float())
                    else:
                        mpd_out_fake, mpd_feat_real, mpd_feat_fake = [], [], []
                    if msd is not None:
                        msd_out_fake, msd_feat_fake = msd(recon_clean.float())
                        msd_out_real, msd_feat_real = msd(wav.float())
                    else:
                        msd_out_fake, msd_feat_real, msd_feat_fake = [], [], []
                    d_fake_list = (mpd_out_fake if mpd is not None else []) + (msd_out_fake if msd is not None else [])
                    if lam_gan_eff > 0.0 and len(d_fake_list) > 0:
                        if args.gan_loss_type == "hinge":
                            gan_g = hinge_g_loss(d_fake_list) * lam_gan_eff
                        else:
                            gan_g = gan_g_loss(d_fake_list) * lam_gan_eff
                    if lam_fm_eff > 0.0 and mpd is not None and msd is not None:
                        fm = (feature_matching_loss(mpd_feat_real, mpd_feat_fake) +
                              feature_matching_loss(msd_feat_real, msd_feat_fake)) * lam_fm_eff


            # NaN guard
            L_mel, _ = finite_or_zero(L_mel, "mel", step, wb, args.nan_guard)
            if isinstance(L_cstft, torch.Tensor):
                L_cstft, _ = finite_or_zero(L_cstft, "cstft", step, wb, args.nan_guard)

            vq_loss, _ = finite_or_zero(vq_loss, "vq_loss", step, wb, args.nan_guard)
            ftp_loss, _ = finite_or_zero(ftp_loss, "ftp", step, wb, args.nan_guard)
            sa_cosine_loss, _ = finite_or_zero(sa_cosine_loss, "sa_cosine", step, wb, args.nan_guard)
            sa_contrast_loss, _ = finite_or_zero(sa_contrast_loss, "sa_contrast", step, wb, args.nan_guard)
            gan_g, _ = finite_or_zero(gan_g, "gan_g", step, wb, args.nan_guard)
            fm, _ = finite_or_zero(fm, "fm", step, wb, args.nan_guard)
            L_mr_stft, _ = finite_or_zero(L_mr_stft, "mr_stft", step, wb, args.nan_guard)


            total_loss = (L_mel + L_ms_mel + L_mr_stft + L_cstft) + vq_loss + ftp_loss + sa_cosine_loss + sa_contrast_loss + gan_g + fm + loss_gumbel

            if not torch.isfinite(total_loss):
                print("[guard] loss is non-finite, skipping micro-batch.")
                bad_micro = True
                break

            if use_scaler:
                scaler.scale(total_loss / accum_steps).backward()
            else:
                (total_loss / accum_steps).backward()

            wav_last = wav.detach()
            recon_last = recon_clean.detach()

            lam_tuple = (lam_ftp, lam_gan_eff, lam_fm_eff)

            last_vals = {
                "loss": total_loss.detach(),
                "L_mel": L_mel.detach(), "L_ms_mel": L_ms_mel.detach(), "L_mr_stft": L_mr_stft.detach(), "L_cstft": L_cstft.detach(),
                "vq_loss": vq_loss.detach(),
                "loss_gumbel": loss_gumbel.detach(),
                "ftp": ftp_loss.detach(),
                "ftp_raw": ftp_raw.detach(),
                "sa_cosine": safe_detach(sa_cosine_loss),
                "sa_contrast": safe_detach(sa_contrast_loss),
                "sa_cosine_raw": safe_detach(sa_cosine_raw),
                "sa_contrast_raw": safe_detach(sa_contrast_raw),
                "gan_g": gan_g.detach(), "fm": fm.detach(),
                "lam": lam_tuple,
                "k_sub": 1,
                "ftp_on": ftp_on,
            }

        if bad_micro:
            opt_enc.zero_grad(set_to_none=True)
            opt_dec.zero_grad(set_to_none=True)
            opt_embed.zero_grad(set_to_none=True)
            bad_micro_streak += 1
            if args.nan_guard and bad_micro_streak >= 8:
                gan_pause_until = step + args.gan_pause_steps
                print(f"[guard] enter SAFE-MODE until step {gan_pause_until} (suppress GAN/FM)")
            continue
        else:
            bad_micro_streak = 0

        # optimizer step
        all_params_embed = [p for p in system.ar.model.parameters() if p.requires_grad] + list(medusa_heads.parameters())
        if use_scaler:
            if args.grad_clip > 0:
                scaler.unscale_(opt_embed)
                torch.nn.utils.clip_grad_norm_(all_params_embed, args.grad_clip)
            scaler.step(opt_embed)

            if step >= args.d_only_steps:
                if args.grad_clip > 0:
                    scaler.unscale_(opt_enc)
                    scaler.unscale_(opt_dec)
                    torch.nn.utils.clip_grad_norm_(params_enc, args.grad_clip)
                    torch.nn.utils.clip_grad_norm_(params_dec, args.grad_clip)
                scaler.step(opt_enc)
                scaler.step(opt_dec)
            else:
                pass  # D-only warmup: skip codec optimizer step

            scaler.update()
        else: # not use_scaler
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(all_params_embed, args.grad_clip)
            opt_embed.step()

            if step >= args.d_only_steps:
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(params_enc, args.grad_clip)
                    torch.nn.utils.clip_grad_norm_(params_dec, args.grad_clip)
                opt_enc.step()
                opt_dec.step()
            else:
                pass  # D-only warmup: skip codec optimizer step

        # Discriminator
        if args.enable_gan and mpd is not None and msd is not None and (last_vals["lam"][1] > 0.0) and (
                wav_last is not None) and (step % args.d_update_every == 0):
            set_requires_grad(mpd, True)
            set_requires_grad(msd, True)
            amp_flag_D = (amp_dtype != torch.float32) if args.gan_amp == "auto" else (args.gan_amp == "amp")
            with torch.autocast(device_type=device.type, enabled=amp_flag_D):
                need_r1 = (args.d_reg_every > 0) and (step % args.d_reg_every == 0)
                real_in = wav_last.detach().float().requires_grad_(True)
                fake_in = recon_last.detach().float()
                mpd_out_real, _ = mpd(real_in)
                mpd_out_fake, _ = mpd(fake_in)
                msd_out_real, _ = msd(real_in)
                msd_out_fake, _ = msd(fake_in)
                
                if args.gan_loss_type == "hinge":
                    d_loss = hinge_d_loss(mpd_out_real + msd_out_real, mpd_out_fake + msd_out_fake)
                else:
                    d_loss = gan_d_loss(mpd_out_real + msd_out_real, mpd_out_fake + msd_out_fake)
            if need_r1:
                with torch.autocast(device_type=device.type, enabled=False):
                    r1 = d_r1_penalty(mpd_out_real + msd_out_real, real_in)
                    d_loss = d_loss + (args.r1_gamma * 0.5) * r1

            d_loss, d_loss_is_nan = finite_or_zero(d_loss, "d_loss", step, wb, args.nan_guard)

            if not d_loss_is_nan:
                opt_d.zero_grad(set_to_none=True)
                if use_scaler:
                    scaler.scale(d_loss).backward()
                    scaler.step(opt_d)
                    scaler.update()
                else:
                    d_loss.backward()
                    opt_d.step()
            elif args.nan_guard:
                print(f"[guard] d_loss is NaN/Inf at step {step}, skipping D-step.")

        # Statistics
        with torch.no_grad():
            def _grad_norm(params):
                total = 0.0
                for p in params:
                    if p.grad is not None:
                        g = p.grad.detach()
                        total += float(g.norm(2).item() ** 2)
                return float(math.sqrt(total))

            gn_embed = _grad_norm([p for p in system.ar.model.parameters() if p.requires_grad])
            gn_enc = _grad_norm(params_enc)
            gn_dec = _grad_norm(params_dec)

            uniq = count_unique_1d(out_main.tokens)
            H = entropy_1d(out_main.tokens, args.n_audio_tokens)
            _ftp_raw = system.ar.loss_next_token(out_main.tokens.clone().contiguous())
            ppl_ftp = float(torch.exp(torch.clamp(_ftp_raw.detach(), max=20.0)).cpu().item()) if torch.isfinite(
                _ftp_raw) else float("inf")
            ppl_slm = 1.0

        # Validation / Loop / Save
        if (step % args.val_every) == 0 and step > 0:
            system.eval()
            v_losses = []
            with torch.no_grad():
                for i, vb in enumerate(va_loader):
                    if i >= 8:
                        break
                    vwav = sanitize_audio(vb["wav"].to(device), clip=args.clip_recon_amp)
                    vout = system(vwav, sr=target_sr)
                    vrec = sanitize_audio(vout.recon, clip=args.clip_recon_amp)
                    vrec = match_length(vrec, vwav.size(-1))
                    with torch.autocast(device_type=device.type, enabled=False):
                        vmel_raw = mel_loss(per_sample_rms_norm(vwav) if args.mel_rms_norm else vwav,
                                            per_sample_rms_norm(vrec) if args.mel_rms_norm else vrec)
                    vmel = vmel_raw * args.lambda_mel
                    vstf = vwav.new_tensor(0.0)
                    v_losses.append((vmel + vstf).item())
                    ensure_dir(args.out_dir)
                    save_wav(f"{args.out_dir}/val_step{step}_sample{i}_codec.wav", vrec[:1], target_sr)
                    wb.log({
                        f"val/audio/codec_{i}": wb.audio(vrec[:1], target_sr, caption=f"val_codec_{i}@{step}"),
                        f"val/mel/codec_{i}": wb.image(
                            mel_image(vrec[0], target_sr, n_fft=1024, hop=target_hop),
                            caption=f"val-mel-codec-{i}@{step}")
                    }, step=step, commit=False)
            system.train()

            # D-only warmup: codec stays in train() mode, no need to re-eval()

            # Lock BN to eval again
            system.codec.model.apply(set_batchnorm_to_eval)

            if v_losses:
                vmean = float(np.mean(v_losses))
                wb.log({"val/rec": vmean}, step=step, commit=False)

        if (step % args.log_every) == 0:
            took = time.time() - t0
            lam_ftp, lam_gan_eff, lam_fm_eff = last_vals["lam"]
            loss_total_safe = float(last_vals["loss"].float().item()) + 1e-8
            share_fm = float(last_vals["fm"].float().item()) / loss_total_safe
            share_cst = float(last_vals["L_cstft"].float().item()
                              if isinstance(last_vals["L_cstft"], torch.Tensor) else 0.0) / loss_total_safe
            if share_fm > args.fm_pause_share:
                gan_pause_until = max(gan_pause_until, step + args.gan_pause_steps)

            print(
                f"[{step}] loss={last_vals['loss'].item():.4f} "
                f"rec_mel={last_vals['L_mel'].item():.4f} "
                f"vq={float(last_vals['vq_loss']):.4f} "
                f"ftp={float(last_vals['ftp']):.4f} ftp_raw={float(last_vals['ftp_raw']):.4f} "
                f"sa_cos={float(last_vals['sa_cosine']):.4f} sa_cos_raw={float(last_vals['sa_cosine_raw']):.4f} "
                f"sa_ctr={float(last_vals['sa_contrast']):.4f} sa_ctr_raw={float(last_vals['sa_contrast_raw']):.4f} "
                f"gan_g={float(last_vals['gan_g']):.4f} fm={float(last_vals['fm']):.4f} "
                f"lam_ftp={lam_ftp:.3f} "
                f"lam_gan_eff={lam_gan_eff:.3f} lam_fm_eff={lam_fm_eff:.3f} "
                f"ftp_on={last_vals['ftp_on']} code_entropy={H:.3f} uniq_codes={uniq} "
                f"gn(embed)={gn_embed:.3e} gn(enc)={gn_enc:.3e} gn(dec)={gn_dec:.3e} "
                f"lr_e={lr_embed:.2e} lr_enc={lr_enc:.2e} lr_dec={lr_dec:.2e} took={took:.2f}s (accum={accum_steps})"
            )
            t0 = time.time()

            wb.log({
                "loss/total": float(last_vals["loss"]),
                "loss/rec_mel": float(last_vals["L_mel"]),
                "loss/ms_mel": float(last_vals["L_ms_mel"]),
                "loss/mr_stft": float(last_vals["L_mr_stft"]),
                "loss/cstft": float(last_vals["L_cstft"]),
                "loss/vq": float(last_vals["vq_loss"]),
                "loss/ftp": float(last_vals["ftp"]),
                "loss/ftp_raw": float(last_vals["ftp_raw"]),
                "loss/sa_cosine": float(last_vals["sa_cosine"]),
                "loss/sa_contrast": float(last_vals["sa_contrast"]),
                "loss/sa_cosine_raw": float(last_vals["sa_cosine_raw"]),
                "loss/sa_contrast_raw": float(last_vals["sa_contrast_raw"]),
                "loss/gan_g": float(last_vals["gan_g"]),
                "loss/fm": float(last_vals["fm"]),
                "scale/lam_ftp": float(lam_ftp),
                "scale/lam_gan_eff": float(lam_gan_eff),
                "scale/lam_fm_eff": float(lam_fm_eff),
                "metrics/ppl_ftp": float(ppl_ftp),
                "codes/entropy": float(H),
                "codes/unique": int(uniq),
                "grad/gn_embed": float(gn_embed),
                "grad/gn_enc": float(gn_enc),
                "grad/gn_dec": float(gn_dec),
                "lr/embed": float(lr_embed),
                "lr/enc": float(lr_enc),
                "lr/dec": float(lr_dec),
                "time/iter_sec": float(max(took, 1e-9)),
                "guard/gan_pause_until": float(gan_pause_until),
                "train/grad_accum_steps": float(accum_steps),
            }, step=step, commit=True)

        if (step % args.save_every) == 0 and step > 0:
            # Write Medusa Head0 weights back to lm_head (audio tokens only)
            # This ensures the saved model is compatible with original structure
            with torch.no_grad():
                system.ar.model.lm_head.weight.data[audio_ids_for_init] = medusa_heads[0].proj.weight.data.clone()
            
            ckpt_path = f"{args.out_dir}/ckpt_step{step}.pt"
            state = {
                "step": step,
                "auv": system.codec.model.state_dict(),
                "qwen_embed": system.ar.model.get_input_embeddings().weight.detach().cpu(),
                "qwen_lm_head": system.ar.model.lm_head.weight.detach().cpu(),  # Include updated lm_head
                "opt_enc": opt_enc.state_dict(),
                "opt_dec": opt_dec.state_dict(),
                "opt_embed": opt_embed.state_dict(),
                "tok_dir": tok_dir,
                "vocab_size": system.ar.model.get_input_embeddings().weight.shape[0],
                "medusa_heads": medusa_heads.state_dict(),  # Save all heads for potential future use
            }
            if args.enable_gan and mpd is not None and msd is not None:
                state["mpd"] = mpd.state_dict()
                state["msd"] = msd.state_dict()
                state["opt_d"] = opt_d.state_dict()
            torch.save(state, ckpt_path)
            wb.log({"ckpt/path": ckpt_path}, step=step, commit=False)


        step += 1

    log_reference_precommit(step - 1)

    # Save final checkpoint (ensures last step is always saved)
    final_step = step - 1
    with torch.no_grad():
        system.ar.model.lm_head.weight.data[audio_ids_for_init] = medusa_heads[0].proj.weight.data.clone()
    
    ckpt_path = f"{args.out_dir}/ckpt_final_step{final_step}.pt"
    state = {
        "step": final_step,
        "auv": system.codec.model.state_dict(),
        "qwen_embed": system.ar.model.get_input_embeddings().weight.detach().cpu(),
        "qwen_lm_head": system.ar.model.lm_head.weight.detach().cpu(),
        "opt_enc": opt_enc.state_dict(),
        "opt_dec": opt_dec.state_dict(),
        "opt_embed": opt_embed.state_dict(),
        "tok_dir": tok_dir,
        "vocab_size": system.ar.model.get_input_embeddings().weight.shape[0],
        "medusa_heads": medusa_heads.state_dict(),
    }
    if args.enable_gan and mpd is not None and msd is not None:
        state["mpd"] = mpd.state_dict()
        state["msd"] = msd.state_dict()
        state["opt_d"] = opt_d.state_dict()
    torch.save(state, ckpt_path)
    print(f"[Checkpoint] Final checkpoint saved: {ckpt_path}")



if __name__ == "__main__":
    main()
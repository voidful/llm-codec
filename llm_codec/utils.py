# -*- coding: utf-8 -*-
"""General Utilities: Files, random seeds, audio I/O, simple stats, mel visualization."""

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio


# --------- Signal Processing Helpers ---------
def per_sample_rms_norm(wav: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-sample RMS Normalization (maintains relative waveform but unifies energy)."""
    rms = torch.sqrt((wav ** 2).mean(dim=-1, keepdim=True) + eps)
    return wav / (rms + eps)


def random_phase_roll(wav: torch.Tensor, max_shift: int) -> torch.Tensor:
    """Random Phase Jitter: randomly roll along time axis to simulate phase changes."""
    if max_shift <= 0:
        return wav
    B = wav.size(0)
    shifts = torch.randint(-max_shift, max_shift + 1, (B,), device=wav.device)
    out = []
    for i in range(B):
        # roll supports int shift, to support different shift per sample needs loop or grid_sample
        # using simple loop here
        out.append(torch.roll(wav[i:i + 1], int(shifts[i].item()), dims=-1))
    return torch.cat(out, dim=0)




# --------- Files and Randomness ---------
def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_wav(path: str, wav: torch.Tensor, sr: int):
    """Accepts (B,1,T) / (1,T) / (T,); saved as mono WAV."""
    wav = wav.detach().float().cpu()
    if wav.ndim == 3:       # (B,1,T)
        wav = wav[0, 0]
    elif wav.ndim == 2:     # (1,T) or (C,T)
        wav = wav[0]
    ensure_dir(Path(path).parent.as_posix())
    torchaudio.save(path, wav.unsqueeze(0), sr)


# --------- Code Statistics ---------
@torch.no_grad()
def count_unique_1d(x: torch.Tensor) -> int:
    """Count unique tokens in 1D/2D integer tensor (flattened to 1D)."""
    return torch.unique(x.reshape(-1)).numel()


@torch.no_grad()
def entropy_1d(x: torch.Tensor, K: int) -> float:
    """Simple entropy estimation with K as upper bound (log normalized to [0,1])."""
    flat = x.reshape(-1)
    hist = torch.bincount(flat, minlength=K).float()
    p = hist / (hist.sum() + 1e-9)
    nz = p[p > 0]
    H = -(nz * torch.log(nz + 1e-12)).sum() / torch.log(torch.tensor(K + 1e-12, dtype=torch.float32))
    return float(H.item())


@torch.no_grad()
def token_topk_table(x: torch.Tensor, k: int = 20) -> List[Tuple[int, int]]:
    """Return top-k (token_id, count) tuple list."""
    flat = x.reshape(-1)
    hist = torch.bincount(flat)
    k = int(min(k, hist.numel()))
    if k <= 0:
        return []
    counts, ids = torch.topk(hist, k)
    return [(int(ids[i].item()), int(counts[i].item())) for i in range(len(counts))]


# --------- Visualization ---------
def mel_image(y: torch.Tensor, sr: int, n_fft=1024, hop=256, n_mels=100) -> np.ndarray:
    """Input single audio sample (1,T) or (T,); output 0~255 mel image (uint8)."""
    if y.ndim == 1:
        y = y.unsqueeze(0)
    elif y.ndim == 2 and y.shape[0] != 1:
        # Take first channel
        y = y[0:1]
    Mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        f_min=0,
        f_max=sr // 2,
        n_mels=n_mels,
        window_fn=torch.hann_window,
        power=2.0,
        center=True,
        pad_mode="reflect",
    ).to(y.device)
    m = Mel(y.float()).clamp_min(1e-6).log().detach().cpu().numpy()
    m = (m - m.min()) / (m.max() - m.min() + 1e-9)
    return (m * 255.0).astype(np.uint8)[0]
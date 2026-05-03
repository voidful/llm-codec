# -*- coding: utf-8 -*-
"""System Wrapper: AUV Codec + Qwen AR combination, unified forward interface."""

from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn

from .codec import AUVCodecWrapper
from .qwen_ar import QwenAR


@dataclass
class SystemOut:
    quant: torch.Tensor  # (B,C,Tf)
    tokens: torch.Tensor  # (B,Tf)
    recon: torch.Tensor  # (B,1,T)
    # --- [!!! Ultimate Fix (VQ Loss) !!!] ---
    vq_loss: torch.Tensor  # <-- Added vq_loss
    # --- [!!! Fix End !!!] ---


class AUVSystem(nn.Module):
    """
    - codec: AUVCodecWrapper (differentiable encode/decode)
    - ar   : QwenAR (only adjusts input embeddings)
    """

    def __init__(
            self,
            ckpt_path: str,
            device: torch.device,
            bf16: bool,
            qwen_model: str,
            mp_dtype: str,
            n_audio_tokens: int,
            ar_token_prefix: str,
            tok_dir: str,
            mean_resizing: bool,
    ):
        super().__init__()
        self.device = device
        self.codec = AUVCodecWrapper(ckpt_path, device=device, bf16=bf16)
        self.ar = QwenAR(
            qwen_model,
            device=device,
            mp_dtype=mp_dtype,
            audio_token_prefix=ar_token_prefix,
            n_audio_tokens=n_audio_tokens,
            tok_dir=tok_dir,
            mean_resizing=mean_resizing,
        )

    @property
    def sample_rate(self) -> int:
        return self.codec.sample_rate

    @property
    def hop_length(self) -> int:
        return self.codec.hop_length

    def forward(self, wav: torch.Tensor, sr: int) -> SystemOut:
        enc = self.codec.encode(wav, sr=sr)
        target_len = wav.shape[-1]
        recon = self.codec.decode(enc["quantized"], target_len=target_len)

        # --- [!!! Ultimate Fix (VQ Loss) !!!] ---
        return SystemOut(
            quant=enc["quantized"],
            tokens=enc["tokens"],
            recon=recon,
            vq_loss=enc["vq_loss"]  # <-- Pass vq_loss out
        )
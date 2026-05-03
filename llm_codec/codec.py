# -*- coding: utf-8 -*-
"""AUV Codec Wrapper: Unified encode/decode interface, handles sample rate and length alignment."""

from __future__ import annotations
from typing import Optional, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class AUVCodecWrapper(nn.Module):
    """Thin wrapper for AUV neural audio codec.

    Goals:
      * Hide @torch.no_grad() on AUV.encode/decode to allow upstream backpropagation.
      * Provide stable batch interface:
          - ``encode(wav, sr) -> {"quantized", "tokens", "vq_loss"}``
          - ``decode(quantized) -> wav``
      * Automatically handle sample rate mismatch and decoder output length deviation to avoid "aliasing" and trailing noise.
    """

    def __init__(
        self,
        ckpt_path: str,
        device: torch.device,
        bf16: bool = True,
        fallback_module_names: Tuple[str, ...] = (
            "token2wav",
            "decoder", "generator", "vocoder", "codec_decoder", "synth", "postnet", "hifigan", "wavernn",
            "net_g",
        ),
        strict_fail_on_no_grad: bool = True,
    ) -> None:
        super().__init__()

        # Dynamically load AUV to avoid crash if environment is missing it
        try:
            from auv.model import AUV
        except ImportError as e:
            raise ImportError("Please install 'auv' package or ensure auv.model.AUV can be imported.") from e

        model = AUV()
        model.from_pretrained(ckpt_path)
        model = model.to(device)

        # AUV official encode/decode are marked with @torch.inference_mode
        # Here we unify to eval mode but keep requires_grad to allow optional fine-tuning.
        model.eval()
        for p in model.parameters():
            p.requires_grad = True

        self.model = model
        self.device = device
        self.autocast_enabled = bool(bf16)
        self._amp_dtype = torch.bfloat16 if bf16 else torch.float32

        # Try to automatically get sample_rate / hop_length
        codec_obj = getattr(self.model, "tokenizer", getattr(self.model, "codec", self.model))
        self.sample_rate = int(getattr(codec_obj, "sample_rate", 16000))
        self.hop_length = int(getattr(codec_obj, "hop_length", 320))

        if not hasattr(self.model, "tokenizer"):
            raise RuntimeError("[AUVCodecWrapper] Cannot find 'model.tokenizer' (Encoder).")
        if not hasattr(self.model, "token2wav"):
            raise RuntimeError("[AUVCodecWrapper] Cannot find 'model.token2wav' (Decoder).")

        self.fallback_module_names = tuple(fallback_module_names)
        self.strict_fail_on_no_grad = bool(strict_fail_on_no_grad)

        # Only warn once when SR mismatch occurs to avoid noisy logs
        self._warned_sr_mismatch: bool = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_tokens(t: torch.Tensor) -> torch.Tensor:
        if not isinstance(t, torch.Tensor):
            t = torch.as_tensor(t)
        if t.dim() == 1:
            t = t.unsqueeze(0)
        elif t.dim() > 2:
            b = t.size(0)
            t = t.view(b, -1)
        return t.long()

    def _decode_via_modules(self, q: torch.Tensor) -> Optional[torch.Tensor]:
        """Fallback decode path.

        Some implementations put decoder under strange member names like ``net_g`` or ``generator``.
        Here we do a simple traversal to find the first module that forwards successfully.
        """
        for name in self.fallback_module_names:
            mod = getattr(self.model, name, None)
            if isinstance(mod, nn.Module):
                try:
                    with torch.enable_grad():
                        out = mod(q)
                    if isinstance(out, (tuple, list)):
                        out = out[0]
                    if isinstance(out, torch.Tensor):
                        return out
                except Exception:
                    continue
        return None

    # ------------------------------------------------------------------
    # encode / decode
    # ------------------------------------------------------------------
    def encode(self, wav: torch.Tensor, sr: int) -> Dict[str, torch.Tensor]:
        """Quantize audio.

        Args:
            wav: Audio tensor with shape ``(B, 1, T)``.
            sr: Current sample rate of wav.

        Returns:
            dict: ``{"quantized": (B, C, Tf), "tokens": (B, Tf), "vq_loss": scalar}``
        """
        if wav.dim() != 3 or wav.size(1) != 1:
            raise ValueError(f"[AUVCodecWrapper] wav must be (B,1,T), got {wav.shape}")

        wav = wav.to(self.device)

        # Resample once here if sample rate mismatches, avoiding upstream handling.
        if int(sr) != int(self.sample_rate):
            try:
                import torchaudio.functional as AF
            except Exception as e:
                raise RuntimeError(
                    f"[AUVCodecWrapper] torchaudio is required to resample from sr={sr} to {self.sample_rate}."
                ) from e

            if not self._warned_sr_mismatch:
                print(
                    f"[AUVCodecWrapper][warn] encode() received sr={sr}, mismatch with codec.sample_rate={self.sample_rate}, "
                    "auto-resampling in wrapper. Future warnings suppressed."
                )
                self._warned_sr_mismatch = True

            wav = AF.resample(wav, orig_freq=sr, new_freq=self.sample_rate)
            sr = self.sample_rate

        with torch.autocast(
            device_type=self.device.type,
            dtype=self._amp_dtype,
            enabled=self.autocast_enabled,
        ), torch.enable_grad():
            out = self.model.tokenizer(wav_input=wav.squeeze(1), sr=sr)

        if not isinstance(out, dict):
            raise RuntimeError(f"[AUVCodecWrapper] tokenizer output must be dict, got {type(out)}")

        if "quantized" not in out or "tokens" not in out:
            raise RuntimeError(
                f"[AUVCodecWrapper] tokenizer output missing 'quantized' or 'tokens' key: {list(out.keys())}"
            )

        feats = out["quantized"]
        toks = out["tokens"]
        vq_loss = out.get("vq_loss", torch.tensor(0.0, device=self.device))

        return {
            "quantized": feats,
            "tokens": self._normalize_tokens(toks),
            "vq_loss": vq_loss,
        }

    def decode(self, quantized: torch.Tensor, target_len: Optional[int] = None) -> torch.Tensor:
        """Decode quantized representation to wave.

        Args:
            quantized: (B, C, Tf). Usually from ``encode()["quantized"]``.
            target_len: Target sample length. If ``None``, inferred from ``Tf * hop_length``.

        Returns:
            wav: (B, 1, T). T will try to align with target_len.
        """
        if not isinstance(quantized, torch.Tensor):
            raise ValueError("[AUVCodecWrapper] quantized must be a Tensor.")

        if quantized.dim() != 3:
            raise ValueError(f"[AUVCodecWrapper] quantized shape must be (B,C,T), got {quantized.shape}")

        quantized = quantized.to(self.device)

        decoder_module = getattr(self.model, "token2wav", None)
        if decoder_module is None:
            rec = self._decode_via_modules(quantized)
            if rec is None:
                raise RuntimeError("[AUVCodecWrapper] No suitable Decoder module found. Please check AUV structure.")
        else:
            with torch.autocast(
                device_type=self.device.type,
                dtype=self._amp_dtype,
                enabled=self.autocast_enabled,
            ), torch.enable_grad():
                rec = decoder_module(quantized)

        if not isinstance(rec, torch.Tensor):
            raise RuntimeError(f"[AUVCodecWrapper] Decoder output type error: {type(rec)}")

        if rec.dim() == 2:
            rec = rec.unsqueeze(0)
        if rec.dim() != 3:
            raise RuntimeError(f"[AUVCodecWrapper] Decoder output shape should be (B,1,T) or (B,T), got {rec.shape}")

        # ---------- Length Alignment: Core Fix ----------
        # AUV decoding sometimes yields length larger than theoretical length by a few hops,
        # which causes trailing noise / aliasing when cropped by Mel/STFT during training.
        # Here we unify cropping to theoretical length.
        if target_len is None and self.hop_length is not None and self.hop_length > 0:
            # Tf comes from quantized time axis
            tf = quantized.shape[-1]
            target_len = int(tf * self.hop_length)

        if target_len is not None and target_len > 0:
            t_cur = rec.size(-1)
            if t_cur > target_len:
                rec = rec[..., :target_len]
            elif t_cur < target_len:
                pad = target_len - t_cur
                rec = F.pad(rec, (0, pad))

        return rec

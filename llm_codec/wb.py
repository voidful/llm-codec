# -*- coding: utf-8 -*-
"""W&B Lightweight Wrapper: Automatically no-op if wandb is not present, ensuring training flow is unaffected."""

from __future__ import annotations
from typing import Optional, Dict, Any
import numpy as np
import torch


class WB:
    def __init__(
        self,
        enabled: bool,
        project: str = "debug",
        run_name: Optional[str] = None,
        dir_: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.enabled = False
        self.wb = None
        self._last_committed = -1
        if enabled:
            try:
                import wandb
                self.wb = wandb
                self.wb.init(
                    project=project,
                    name=run_name,
                    dir=dir_,
                    config=config,
                    save_code=False,
                )
                self.enabled = True
            except Exception as e:
                print(f"[wandb] Failed to init (continuing as no-op): {e}")

    def log(self, data: Dict[str, Any], step: Optional[int] = None, commit: bool = False):
        if not self.enabled:
            return
        try:
            # Fix error caused by step value regression
            if step is not None and self._last_committed >= 0 and step < self._last_committed:
                step = self._last_committed
            if step is None:
                self.wb.log(data, commit=commit)
            else:
                self.wb.log(data, step=step, commit=commit)
            if commit:
                self._last_committed = max(self._last_committed, step if step is not None else self._last_committed + 1)
        except Exception as e:
            print(f"[wandb] log failed: {e}")

    def watch(self, model, log: str = "gradients", log_freq: int = 1000):
        if self.enabled:
            try:
                self.wb.watch(model, log=log, log_freq=log_freq)
            except Exception:
                pass

    def audio(self, wav: torch.Tensor, sr: int, caption: str):
        """Convert tensor audio to W&B Audio object; return None if no-op."""
        if not self.enabled:
            return None
        # Supports (B,1,T) / (1,T) / (B,C,T)
        if wav.ndim == 3:
            # Take first channel of first batch
            wav = wav[0, 0]
        elif wav.ndim == 2 and wav.shape[0] == 1:
            wav = wav[0]
        arr = wav.detach().cpu().float().numpy()
        try:
            return self.wb.Audio(arr, sample_rate=sr, caption=caption)
        except Exception:
            return None

    def image(self, img_np: np.ndarray, caption: str = ""):
        """Image logging (mel spect, etc.); return None if no-op."""
        if not self.enabled:
            return None
        try:
            return self.wb.Image(img_np, caption=caption)
        except Exception:
            return None
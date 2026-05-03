# -*- coding: utf-8 -*-
"""Tools for Learning Rate, Weight, Curriculum, and Time Alignment."""

from __future__ import annotations
import math
import torch
import torch.nn.functional as F


# ---- LR / Lambda Scheduling ----
def cosine_with_warmup(step: int, base_lr: float, warmup: int, total: int) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    t = min(max(t, 0.0), 1.0)
    return 0.5 * base_lr * (1 + math.cos(math.pi * t))


def lambda_ramp(step: int, warmup: int, ramp: int, target: float) -> float:
    if step < warmup:
        return 0.0
    if ramp <= 0:
        return target
    t = (step - warmup) / float(max(1, ramp))
    t = max(0.0, min(1.0, t))
    return target * t


def cosine_ramp(step: int, warmup: int, ramp: int, target: float) -> float:
    if step < warmup:
        return 0.0
    if ramp <= 0:
        return target
    t = (step - warmup) / float(max(1, ramp))
    t = max(0.0, min(1.0, t))
    return target * 0.5 * (1 - math.cos(math.pi * t))


def cosine_decay(step: int, init: float, final: float, steps: int) -> float:
    if steps <= 0:
        return final
    t = max(0.0, min(1.0, step / float(steps)))
    return final + (init - final) * 0.5 * (1 + math.cos(math.pi * t))


# ---- AR Curriculum (Subsampling) ----
def ar_curriculum_k(
    step: int,
    schedule: str,
    k4_until_h2e: int,
    k2_until_h2e: int,
    k1_until_e2h: int,
    k2_until_e2h: int,
) -> int:
    """Determine current subsampling factor k ∈ {1,2,4}."""
    if schedule == "easy2hard":
        if step < k1_until_e2h:
            return 1
        if step < k2_until_e2h:
            return 2
        return 4
    # Default hard2easy
    if step < k4_until_h2e:
        return 4
    if step < k2_until_h2e:
        return 2
    return 1


def subsample_codes(codes: torch.Tensor, k: int) -> torch.Tensor:
    """Subsample sequence with stride k (return directly if k<=1)."""
    if k <= 1:
        return codes
    return codes[:, ::k].contiguous()


def align_BTV_to_T(logits_B_T_V: torch.Tensor, target_T: int) -> torch.Tensor:
    """Linear interpolation alignment (B,Tq,V) -> Target Length T."""
    B, Tq, V = logits_B_T_V.shape
    if Tq == target_T:
        return logits_B_T_V
    x = logits_B_T_V.transpose(1, 2)  # (B,V,Tq)
    x = F.interpolate(x, size=target_T, mode="linear", align_corners=False)
    return x.transpose(1, 2).contiguous()
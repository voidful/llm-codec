# -*- coding: utf-8 -*-
"""GAN Structure (MPD/MSD) and corresponding losses, plus D requires_grad control during training."""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---- Initialization ----
def weights_init(m: nn.Module):
    if isinstance(m, (nn.Conv1d, nn.Conv2d)):
        nn.init.normal_(m.weight, 0.0, 0.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# ---- MPD: Multi-Period Discriminator ----
class DiscriminatorP(nn.Module):
    """Reshape time dimension to 2D with period p, perform 2D convolution discrimination."""
    def __init__(self, period: int):
        super().__init__()
        self.period = period
        chs = [1, 64, 128, 256, 512, 1024]
        layers = []
        for i in range(len(chs) - 1):
            in_c, out_c = chs[i], chs[i + 1]
            layers += [
                nn.Conv2d(in_c, out_c, kernel_size=(5, 1), stride=(3 if i > 0 else 1, 1), padding=(2, 0)),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        layers += [
            nn.Conv2d(chs[-1], chs[-1], kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        self.convs = nn.Sequential(*layers)
        self.out = nn.Conv2d(chs[-1], 1, kernel_size=(3, 1), padding=(1, 0))
        self.apply(weights_init)

    def forward(self, x: torch.Tensor):
        # x: (B,1,T) -> reshape to (B,1,T//p,p) then specific 2D conv
        b, c, t = x.shape
        if t % self.period != 0:
            pad = self.period - (t % self.period)
            x = F.pad(x, (0, pad), mode="reflect")
            t = t + pad
        x = x.view(b, c, t // self.period, self.period)
        feats = []
        h = x
        for layer in self.convs:
            h = layer(h)
            if isinstance(layer, nn.LeakyReLU):
                feats.append(h)
        out = self.out(h)
        return out, feats


class MPD(nn.Module):
    def __init__(self, periods=(2, 3, 5, 7, 11)):
        super().__init__()
        self.discriminators = nn.ModuleList([DiscriminatorP(p) for p in periods])

    def forward(self, x: torch.Tensor):
        outs, feats = [], []
        for d in self.discriminators:
            o, f = d(x)
            outs.append(o)
            feats.append(f)
        return outs, feats


# ---- MSD: Multi-Scale Discriminator ----
class SubDiscriminator(nn.Module):
    """1D Multi-Layer Convolution Discriminator (as MSD subnet)."""
    def __init__(self):
        super().__init__()
        chs = [1, 64, 128, 256, 512, 1024]
        layers = []
        for i in range(len(chs) - 1):
            k = 15 if i == 0 else 41
            s = 1 if i == 0 else 2
            p = (k - 1) // 2
            layers += [
                nn.Conv1d(chs[i], chs[i + 1], kernel_size=k, stride=s, padding=p),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        layers += [
            nn.Conv1d(chs[-1], chs[-1], kernel_size=5, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        self.convs = nn.Sequential(*layers)
        self.out = nn.Conv1d(chs[-1], 1, kernel_size=3, padding=1)
        self.apply(weights_init)

    def forward(self, x: torch.Tensor):
        feats = []
        h = x
        for layer in self.convs:
            h = layer(h)
            if isinstance(layer, nn.LeakyReLU):
                feats.append(h)
        out = self.out(h)
        return out, feats


class MSD(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool1d(4, 2, padding=1, count_include_pad=False)
        self.pool2 = nn.AvgPool1d(4, 2, padding=1, count_include_pad=False)
        self.discriminators = nn.ModuleList([SubDiscriminator(), SubDiscriminator(), SubDiscriminator()])

    def forward(self, x: torch.Tensor):
        outs, feats = [], []
        h = x
        for i, d in enumerate(self.discriminators):
            o, f = d(h)
            outs.append(o)
            feats.append(f)
            if i == 0:
                h = self.pool(h)
            elif i == 1:
                h = self.pool2(h)
        return outs, feats


# ---- GAN Losses and Tools ----
def gan_d_loss(d_real, d_fake):
    loss = 0.0
    for pr, pf in zip(d_real, d_fake):
        loss = loss + F.mse_loss(pr, torch.ones_like(pr)) + F.mse_loss(pf, torch.zeros_like(pf))
    return loss


def hinge_d_loss(d_real, d_fake):
    loss = 0.0
    for pr, pf in zip(d_real, d_fake):
        loss = loss + torch.mean(F.relu(1.0 - pr)) + torch.mean(F.relu(1.0 + pf))
    return loss


def hinge_g_loss(d_fake):
    loss = 0.0
    for pf in d_fake:
        loss = loss - torch.mean(pf)
    return loss


def gan_g_loss(d_fake):
    loss = 0.0
    for pf in d_fake:
        loss = loss + F.mse_loss(pf, torch.ones_like(pf))
    return loss


def feature_matching_loss(feats_real, feats_fake):
    """
    feats_real / feats_fake:
      - Feature list from MPD / MSD
      - Structure usually: list[discriminator][list[feature_map (Tensor)]]

    To avoid feature dimension mismatch due to different input lengths of real / fake,
    we crop both to the common minimum length before calculating L1.
    """
    loss = 0.0
    for fr_list, ff_list in zip(feats_real, feats_fake):
        for fr, ff in zip(fr_list, ff_list):
            # If shapes match exactly, compute directly
            if fr.shape == ff.shape:
                loss = loss + F.l1_loss(fr, ff)
                continue

            # Align to common minimum size (min across all dims, batch/channel are usually same)
            # Typically differences are only in time dimension (or freq dimension).
            ndim = fr.dim()
            slices = []
            for d in range(ndim):
                m = min(fr.size(d), ff.size(d))
                slices.append(slice(0, m))

            fr_c = fr[tuple(slices)]
            ff_c = ff[tuple(slices)]

            loss = loss + F.l1_loss(fr_c, ff_c)
    return loss



def set_requires_grad(module: nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad_(flag)
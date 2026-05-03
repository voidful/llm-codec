# -*- coding: utf-8 -*-
"""Qwen AR: Only finetune input embeddings, mapping discrete audio codes to extended vocabulary."""

from __future__ import annotations
import os
import json
from pathlib import Path
from typing import Optional, List

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM


class QwenAR:
    """
    - Add massive audio special tokens (e.g., <CODEC_0>, <CODEC_1>, ...) to tokenizer.
    - Only expose LLM input embedding parameters with requires_grad=True.
    - Provides:
        .codes_to_ids(codes)   -> ids
        .loss_next_token(codes) -> Standard autoregressive NLL (using codes themselves as learning target)
        .next_token_logits(codes) -> Stepwise logits (for KL alignment)
    """
    def __init__(
        self,
        model_name: str,
        device: torch.device,
        mp_dtype: str = "bf16",
        audio_token_prefix: str = "<CODEC_",
        n_audio_tokens: int = 20480,
        tok_dir: Optional[str] = None,
        mean_resizing: bool = False,
    ):
        self.device = device
        self.n_audio_tokens = int(n_audio_tokens)
        self.audio_tokens: List[str] = [f"{audio_token_prefix}{i}>" for i in range(self.n_audio_tokens)]
        self.tok_dir = tok_dir

        # ---- Tokenizer Setup (Supports persistence) ----
        tok = None
        if tok_dir:
            tok_path = Path(tok_dir)
            if tok_path.exists() and (tok_path / "tokenizer_config.json").exists():
                try:
                    tok = AutoTokenizer.from_pretrained(tok_dir, trust_remote_code=True)
                    print(f"[QwenAR] Loaded tokenizer from {tok_dir}.")
                except Exception as e:
                    print(f"[QwenAR][warn] Failed to load {tok_dir}: {e}")

        if tok is None:
            tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            to_add = [t for t in self.audio_tokens if t not in tok.get_vocab()]
            added = tok.add_tokens(to_add)
            print(f"[QwenAR] Added {added} audio tokens (target {self.n_audio_tokens}).")
            if tok_dir:
                os.makedirs(tok_dir, exist_ok=True)
                tok.save_pretrained(tok_dir)
                with open(os.path.join(tok_dir, "audio_tokens_meta.json"), "w", encoding="utf-8") as f:
                    json.dump({"prefix": audio_token_prefix, "n_audio_tokens": self.n_audio_tokens}, f, ensure_ascii=False, indent=2)
        else:
            missing = [t for t in self.audio_tokens if t not in tok.get_vocab()]
            if missing:
                added = tok.add_tokens(missing)
                print(f"[QwenAR] Added {added} missing audio tokens to existing tokenizer.")
                if tok_dir:
                    tok.save_pretrained(tok_dir)

        self.tok = tok

        # ---- Model & Dtype ----
        if mp_dtype == "bf16":
            dtype = torch.bfloat16
        elif mp_dtype == "fp16":
            dtype = torch.float16
        else:
            dtype = torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=None,
        ).to(device)

        # ---- Expand embeddings to tokenizer size (supporting both old and new mean_resizing param) ----
        want_vocab = len(self.tok)
        emb_module = self.model.get_input_embeddings()
        cur_vocab = emb_module.num_embeddings
        emb_path = os.path.join(tok_dir if tok_dir else ".", "embeddings.pt")

        def _save_embeddings():
            if not tok_dir:
                return
            os.makedirs(tok_dir, exist_ok=True)
            torch.save(
                {
                    "weight": emb_module.weight.detach().cpu(),
                    "num_embeddings": emb_module.num_embeddings,
                    "embedding_dim": emb_module.embedding_dim,
                    "model_name": model_name,
                },
                emb_path,
            )
            self.tok.save_pretrained(tok_dir)
            print(f"[QwenAR] Saved resized embeddings to {emb_path} (vocab={emb_module.num_embeddings}).")

        def _load_embeddings_if_match() -> bool:
            if not os.path.isfile(emb_path):
                return False
            try:
                pkg = torch.load(emb_path, map_location="cpu")
                w = pkg["weight"]
                if w.shape == emb_module.weight.shape and pkg.get("num_embeddings", w.shape[0]) == want_vocab:
                    with torch.no_grad():
                        emb_module.weight.copy_(w)
                    print(f"[QwenAR] Loaded embeddings.pt directly (skipped resize), shape={tuple(w.shape)}")
                    return True
            except Exception as e:
                print(f"[QwenAR][warn] Failed to load embeddings: {e}")
            return False

        if cur_vocab == want_vocab:
            if not _load_embeddings_if_match():
                _save_embeddings()
        else:
            print(f"[QwenAR] resize_token_embeddings: {cur_vocab} -> {want_vocab} (mean_resizing={mean_resizing})")
            try:
                # New transformers support mean_resizing
                self.model.resize_token_embeddings(want_vocab, mean_resizing=bool(mean_resizing))
            except TypeError:
                # Old version fallback
                self.model.resize_token_embeddings(want_vocab)
            emb_module = self.model.get_input_embeddings()
            if not _load_embeddings_if_match():
                _save_embeddings()

        # ---- Train Input Embeddings Only ----
        self.model.tie_weights()
        for _, p in self.model.named_parameters():
            p.requires_grad = False
        for n, p in self.model.named_parameters():
            # Common name override (embed_tokens/word_embeddings/wte)
            if "embed_tokens" in n or "wte" in n or "word_embeddings" in n:
                p.requires_grad = True
        self.model.train()

        # Build audio token ID table
        ids = self.tok.convert_tokens_to_ids(self.audio_tokens)
        self.audio_token_id_table = torch.tensor(ids, device=device, dtype=torch.long)

    # ---- Interface ----
    @property
    def vocab_size(self) -> int:
        return int(self.model.get_input_embeddings().num_embeddings)

    def codes_to_ids(self, codes: torch.Tensor) -> torch.Tensor:
        """Map discrete codes (0..N-1) to tokenizer token_ids."""
        if not isinstance(codes, torch.Tensor):
            codes = torch.as_tensor(codes)
        codes = codes.long().contiguous()
        if codes.dim() == 1:
            codes = codes.unsqueeze(0)
        elif codes.dim() >= 3:
            B = codes.size(0)
            codes = codes.view(B, -1)
        codes = codes.clamp_(0, self.audio_token_id_table.numel() - 1)
        ids = self.audio_token_id_table[codes]
        return ids

    def loss_next_token(self, codes: torch.Tensor) -> torch.Tensor:
        """Autoregressive next-token cross-entropy (depends on input embeddings only)."""
        ids = self.codes_to_ids(codes)
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        elif ids.dim() >= 3:
            ids = ids.view(ids.size(0), -1)
        B, T = ids.shape
        if T < 2:
            return ids.new_tensor(0.0, dtype=torch.float32)
        bos_id = self.tok.bos_token_id
        if bos_id is not None:
            inp = torch.cat([torch.full((B, 1), bos_id, device=ids.device, dtype=ids.dtype), ids[:, :-1]], dim=1)
            labels = ids
        else:
            inp = ids[:, :-1]
            labels = ids[:, 1:]
        out = self.model(input_ids=inp, labels=labels, use_cache=False)
        return out.loss

    @torch.no_grad()
    def next_token_logits(self, codes: torch.Tensor) -> torch.Tensor:
        """Return stepwise logits (for KL alignment). shape = (B, T, V)"""
        ids = self.codes_to_ids(codes)
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        B, T = ids.shape
        bos_id = self.tok.bos_token_id
        if bos_id is not None:
            inp = torch.cat([torch.full((B, 1), bos_id, device=ids.device, dtype=ids.dtype), ids[:, :-1]], dim=1)
            logits = self.model(input_ids=inp, use_cache=False).logits  # (B,T,V)
            return logits
        else:
            inp = ids[:, :-1]
            logits = self.model(input_ids=inp, use_cache=False).logits  # (B,T-1,V)
            pad = torch.zeros(B, 1, logits.size(-1), device=logits.device, dtype=logits.dtype)
            return torch.cat([pad, logits], dim=1)
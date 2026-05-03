"""Extract model artifacts from a training checkpoint and upload to HuggingFace Hub.

If the checkpoint's tokenizer uses the legacy <AUV_*> or <SPEECH_*> audio token
prefix, this script automatically migrates them to the canonical <CODEC_*> prefix
before pushing to the Hub.

Usage:
    python extract_upload.py --ckpt ./runs/.../ckpt_step20000.pt \\
                             --repo_id voidful/llm-codec \\
                             [--qwen_model Qwen/Qwen3-4B-Instruct-2507]
"""

import argparse
import os
import re
import shutil
import tempfile

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import HfApi


# ---- Legacy → Canonical prefix migration ----
LEGACY_PREFIXES = ["<AUV_", "<SPEECH_"]
CANONICAL_PREFIX = "<CODEC_"

# Pattern: matches <AUV_123> or <SPEECH_456> etc.
_LEGACY_RE = re.compile(r"<(?:AUV|SPEECH)_(\d+)>")


def _detect_legacy_prefix(tokenizer) -> str | None:
    """Return the legacy prefix if found, else None."""
    vocab = tokenizer.get_vocab()
    for pfx in LEGACY_PREFIXES:
        if f"{pfx}0>" in vocab:
            return pfx
    return None


def _count_prefix_tokens(tokenizer, prefix: str) -> int:
    """Count consecutive tokens <prefix0>, <prefix1>, ..."""
    vocab = tokenizer.get_vocab()
    n = 0
    while f"{prefix}{n}>" in vocab:
        n += 1
    return n


def migrate_tokenizer(tokenizer, old_prefix: str, n_audio_tokens: int, tok_dir: str,
                      base_model_name: str = "Qwen/Qwen3-4B-Instruct-2507"):
    """Replace legacy audio tokens with <CODEC_*> tokens.

    Strategy: build a CLEAN tokenizer from the base model, add only <CODEC_*>,
    and save.  The old approach (in-place rename) left <AUV_*> residues in the
    vocab because HF tokenizer files have multiple redundant token lists.
    """
    print(f"[Migrate] Building clean tokenizer with {n_audio_tokens} "
          f"{CANONICAL_PREFIX}* tokens (replacing {old_prefix}*)")

    # Record old IDs for verification
    old_first_id = tokenizer.convert_tokens_to_ids(f"{old_prefix}0>")

    # Build a FRESH tokenizer from the base model
    clean_tok = AutoTokenizer.from_pretrained(base_model_name, use_fast=True,
                                               trust_remote_code=True)
    # Add ONLY canonical <CODEC_*> tokens
    codec_tokens = [f"{CANONICAL_PREFIX}{i}>" for i in range(n_audio_tokens)]
    added = clean_tok.add_tokens(codec_tokens)
    print(f"[Migrate] Added {added} {CANONICAL_PREFIX}* tokens to clean tokenizer")

    # Save
    os.makedirs(tok_dir, exist_ok=True)
    clean_tok.save_pretrained(tok_dir)

    # Reload and verify
    migrated = AutoTokenizer.from_pretrained(tok_dir, trust_remote_code=True)
    new_first_id = migrated.convert_tokens_to_ids(f"{CANONICAL_PREFIX}0>")

    # Verify no legacy tokens remain
    for pfx in LEGACY_PREFIXES:
        if f"{pfx}0>" in migrated.get_vocab():
            raise RuntimeError(f"Migration failed: legacy {pfx}* tokens still in vocab!")

    n_check = _count_prefix_tokens(migrated, CANONICAL_PREFIX)
    assert n_check == n_audio_tokens, (
        f"Migration failed: expected {n_audio_tokens} CODEC tokens, got {n_check}")

    print(f"[Migrate] ✓ Clean tokenizer: {CANONICAL_PREFIX}0> = id {new_first_id}, "
          f"total audio = {n_check}, vocab_size = {len(migrated)}")
    print(f"[Migrate]   (old {old_prefix}0> was id {old_first_id}; "
          f"IDs may differ — embeddings will be remapped in prepare_model)")

    return migrated


def load_checkpoint(path: str) -> dict:
    return torch.load(path, map_location="cpu")


def prepare_auv(ckpt: dict, output_path: str = "llm-codec.pt"):
    """Save the AUV codec weights as a standalone file."""
    torch.save(ckpt["auv"], output_path)
    print(f"[AUV] Saved codec weights to {output_path}")
    return output_path


def prepare_model(ckpt: dict, qwen_model: str = "Qwen/Qwen3-4B-Instruct-2507"):
    """Load Qwen model, apply checkpoint embeddings, and migrate tokenizer if needed.

    Handles three scenarios:
    1. Tokenizer has only <AUV_*>  → migrate to <CODEC_*>, remap embeddings
    2. Tokenizer has both <CODEC_*> AND <AUV_*> → purge AUV, keep CODEC embeddings
    3. Tokenizer has only <CODEC_*> → no migration needed
    """
    tok_dir = ckpt.get("tok_dir", None)
    if tok_dir is None:
        raise ValueError("Checkpoint missing 'tok_dir' key")

    old_tokenizer = AutoTokenizer.from_pretrained(tok_dir, trust_remote_code=True)
    old_embed = ckpt["qwen_embed"]  # (old_vocab_size, hidden)
    old_vocab_size, hidden_dim = old_embed.shape

    legacy_prefix = _detect_legacy_prefix(old_tokenizer)
    has_codec = f"{CANONICAL_PREFIX}0>" in old_tokenizer.get_vocab()

    if legacy_prefix and has_codec:
        # ---- Case 2: BOTH exist (the duplicate bug) ----
        n_codec = _count_prefix_tokens(old_tokenizer, CANONICAL_PREFIX)
        n_auv = _count_prefix_tokens(old_tokenizer, legacy_prefix)
        codec_start = old_tokenizer.convert_tokens_to_ids(f"{CANONICAL_PREFIX}0>")
        auv_start = old_tokenizer.convert_tokens_to_ids(f"{legacy_prefix}0>")

        print(f"[Info] DUPLICATE AUDIO TOKENS detected!")
        print(f"  {CANONICAL_PREFIX}*: {n_codec} tokens, start_id={codec_start}")
        print(f"  {legacy_prefix}*: {n_auv} tokens, start_id={auv_start}")
        print(f"  Old vocab: {old_vocab_size}, will shrink to {old_vocab_size - n_auv}")

        # Build clean tokenizer
        n_audio = n_codec  # use the <CODEC_*> count
        migrate_dir = os.path.join(tok_dir, "_migrated")
        clean_tokenizer = migrate_tokenizer(
            old_tokenizer, legacy_prefix, n_audio, migrate_dir,
            base_model_name=qwen_model)
        new_codec_start = clean_tokenizer.convert_tokens_to_ids(f"{CANONICAL_PREFIX}0>")

        # Remap embeddings: take <CODEC_*> rows from old embeddings
        # Old layout: [base_vocab | <CODEC_0..N> | <AUV_0..N>]
        # New layout: [base_vocab | <CODEC_0..N>]
        new_vocab_size = len(clean_tokenizer)
        new_embed = torch.zeros(new_vocab_size, hidden_dim, dtype=old_embed.dtype)

        # Copy base vocab embeddings
        base_end = min(codec_start, new_codec_start, new_vocab_size)
        new_embed[:base_end] = old_embed[:base_end]

        # Copy CODEC embeddings from old positions to new positions
        for i in range(n_audio):
            old_id = codec_start + i
            new_id = new_codec_start + i
            if old_id < old_vocab_size and new_id < new_vocab_size:
                new_embed[new_id] = old_embed[old_id]

        print(f"[Info] Remapped {n_audio} CODEC embeddings: "
              f"old [{codec_start}:{codec_start+n_audio}] → "
              f"new [{new_codec_start}:{new_codec_start+n_audio}]")
        print(f"[Info] Discarded {n_auv} AUV embeddings at [{auv_start}:{auv_start+n_auv}]")

        # Copy migrated files back
        for fname in os.listdir(migrate_dir):
            shutil.copy2(os.path.join(migrate_dir, fname), os.path.join(tok_dir, fname))
        shutil.rmtree(migrate_dir)
        tokenizer = AutoTokenizer.from_pretrained(tok_dir, trust_remote_code=True)

    elif legacy_prefix and not has_codec:
        # ---- Case 1: Only <AUV_*> (simple migration) ----
        n_audio = _count_prefix_tokens(old_tokenizer, legacy_prefix)
        auv_start = old_tokenizer.convert_tokens_to_ids(f"{legacy_prefix}0>")

        migrate_dir = os.path.join(tok_dir, "_migrated")
        clean_tokenizer = migrate_tokenizer(
            old_tokenizer, legacy_prefix, n_audio, migrate_dir,
            base_model_name=qwen_model)
        new_codec_start = clean_tokenizer.convert_tokens_to_ids(f"{CANONICAL_PREFIX}0>")

        new_vocab_size = len(clean_tokenizer)
        new_embed = torch.zeros(new_vocab_size, hidden_dim, dtype=old_embed.dtype)

        base_end = min(auv_start, new_codec_start, new_vocab_size)
        new_embed[:base_end] = old_embed[:base_end]

        for i in range(n_audio):
            old_id = auv_start + i
            new_id = new_codec_start + i
            if old_id < old_vocab_size and new_id < new_vocab_size:
                new_embed[new_id] = old_embed[old_id]

        print(f"[Info] Remapped {n_audio} embeddings: "
              f"AUV [{auv_start}:{auv_start+n_audio}] → "
              f"CODEC [{new_codec_start}:{new_codec_start+n_audio}]")

        for fname in os.listdir(migrate_dir):
            shutil.copy2(os.path.join(migrate_dir, fname), os.path.join(tok_dir, fname))
        shutil.rmtree(migrate_dir)
        tokenizer = AutoTokenizer.from_pretrained(tok_dir, trust_remote_code=True)

    else:
        # ---- Case 3: Already clean <CODEC_*> only ----
        n_audio = _count_prefix_tokens(old_tokenizer, CANONICAL_PREFIX)
        print(f"[Info] Tokenizer already uses {CANONICAL_PREFIX}* prefix ({n_audio} tokens)")
        tokenizer = old_tokenizer
        new_embed = old_embed

    new_vocab_size = len(tokenizer)

    # Load base model and apply embeddings
    model = AutoModelForCausalLM.from_pretrained(qwen_model, trust_remote_code=True)
    model.resize_token_embeddings(new_vocab_size)
    model.config.vocab_size = new_vocab_size

    with torch.no_grad():
        new_embed = new_embed.to(
            model.model.embed_tokens.weight.device,
            dtype=model.model.embed_tokens.weight.dtype,
        )
        model.model.embed_tokens.weight.copy_(new_embed)
        model.lm_head.weight = model.model.embed_tokens.weight

    # Final sanity check
    v = tokenizer.get_vocab()
    for pfx in LEGACY_PREFIXES:
        if f"{pfx}0>" in v:
            print(f"[WARNING] Legacy {pfx}* tokens still present in final tokenizer!")

    print(f"[Info] Final model: vocab_size={new_vocab_size}, "
          f"embed_shape={tuple(model.model.embed_tokens.weight.shape)}")
    return tokenizer, model


def push_artifacts(tokenizer, model, repo_id: str, codec_pt_path: str = "llm-codec.pt"):
    """Push tokenizer, model, and codec weights to HuggingFace Hub."""
    print(f"[Push] Uploading to {repo_id}...")

    api = HfApi()

    # --- Clean up stale files from previous uploads ---
    # push_to_hub() only adds/updates files; it never deletes orphans.
    # This leaves behind e.g. old multi-shard model files or added_tokens.json
    # with legacy <AUV_*> tokens that poison the tokenizer on load.
    try:
        existing_files = set(api.list_repo_files(repo_id))

        # 1. Delete stale added_tokens.json (legacy <AUV_*> tokens live here)
        #    The fast tokenizer (tokenizer.json) is self-contained; added_tokens.json
        #    is redundant but AutoTokenizer merges it on load, re-introducing AUV.
        if "added_tokens.json" in existing_files:
            print("[Cleanup] Deleting stale added_tokens.json from HF repo")
            api.delete_file("added_tokens.json", repo_id, repo_type="model",
                            commit_message="Remove stale added_tokens.json with legacy <AUV_*> tokens")

        # 2. Delete old multi-shard model files (if we now push a single shard)
        old_shards = [f for f in existing_files
                      if re.match(r"model-\d+-of-\d+\.safetensors", f)]
        if old_shards:
            print(f"[Cleanup] Deleting {len(old_shards)} old model shard files")
            for shard in old_shards:
                api.delete_file(shard, repo_id, repo_type="model",
                                commit_message=f"Remove stale shard {shard}")
        if "model.safetensors.index.json" in existing_files:
            api.delete_file("model.safetensors.index.json", repo_id, repo_type="model",
                            commit_message="Remove stale shard index")

    except Exception as e:
        print(f"[Cleanup] Warning: cleanup failed ({e}), continuing with push...")

    tokenizer.push_to_hub(repo_id)
    model.push_to_hub(repo_id)
    api.upload_file(
        path_or_fileobj=codec_pt_path,
        path_in_repo="llm-codec.pt",
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"[Push] ✓ All artifacts uploaded to {repo_id}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract model from checkpoint and upload to HuggingFace Hub"
    )
    parser.add_argument("--ckpt", type=str,
                        default="./runs/llm_codec/ckpt_final_step24999.pt",
                        help="Path to training checkpoint")
    parser.add_argument("--repo_id", type=str,
                        default="voidful/llm-codec",
                        help="HuggingFace Hub repository ID")
    parser.add_argument("--qwen_model", type=str,
                        default="Qwen/Qwen3-4B-Instruct-2507",
                        help="Base Qwen model name")
    args = parser.parse_args()
    
    print(f"[Info] Loading checkpoint: {args.ckpt}")
    ckpt = load_checkpoint(args.ckpt)
    
    codec_pt = prepare_auv(ckpt)
    tokenizer, model = prepare_model(ckpt, args.qwen_model)
    push_artifacts(tokenizer, model, args.repo_id, codec_pt)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Synthesize webpage comparison audio with the Codec-SUPERB SoundCodec API.

The script samples Codec-SUPERB-tiny Speech/Music/Audio splits, writes the
ground-truth clips, then reconstructs each clip with the paper baselines when
they are available in the local SoundCodec registry.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

codec_superb_root = os.environ.get("CODEC_SUPERB_ROOT", "/tmp/Codec-SUPERB")
if Path(codec_superb_root, "SoundCodec").is_dir():
    sys.path.insert(0, codec_superb_root)

if os.environ.get("LLMCODEC_WEBPAGE_DEVICE", "auto").lower() == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
import torchaudio
from datasets import load_dataset
from SoundCodec import codec as codec_registry


DOMAIN_CONFIG = {
    "speech": {"split": "Speech", "prefix": "s", "out_subdir": "speech"},
    "music": {"split": "Music", "prefix": "m", "out_subdir": "music"},
    "env": {"split": "Audio", "prefix": "e", "out_subdir": "env"},
}

DEFAULT_CODECS = [
    ("llmcodec", "LLM-Codec", "llmcodec"),
    ("auv", "AUV", "auv"),
    ("bigcodec", "BigCodec", "bigcodec_1k"),
    ("unicodec", "UniCodec", "unicodec_24k"),
    ("wavtok", "WavTokenizer-L", "wavtokenizer_24k_large_600_4096"),
]


def _mono_float32(array) -> np.ndarray:
    wav = np.asarray(array, dtype=np.float32)
    if wav.ndim == 2:
        wav = wav.mean(axis=0) if wav.shape[0] <= wav.shape[1] else wav.mean(axis=1)
    wav = np.nan_to_num(wav, nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    if peak > 1.0:
        wav = wav / peak
    return wav


def _fit_seconds(wav: np.ndarray, sample_rate: int, seconds: float) -> np.ndarray:
    if seconds <= 0:
        return wav
    target = int(round(sample_rate * seconds))
    if target <= 0:
        return wav
    if wav.shape[-1] > target:
        return wav[:target]
    if wav.shape[-1] < target:
        pad = target - wav.shape[-1]
        return np.pad(wav, (0, pad))
    return wav


def _save_wav(path: Path, wav: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor = torch.from_numpy(_mono_float32(wav)).unsqueeze(0)
    torchaudio.save(str(path), tensor, int(sample_rate))


def _audio_from_example(example: dict, seconds: float) -> Tuple[np.ndarray, int]:
    audio = example["audio"]
    sample_rate = int(audio["sampling_rate"])
    wav = _mono_float32(audio["array"])
    wav = _fit_seconds(wav, sample_rate, seconds)
    return wav, sample_rate


def _iter_examples(dataset_name: str, split: str, count: int, streaming: bool, cache_dir: str | None):
    ds = load_dataset(dataset_name, split=split, streaming=streaming, cache_dir=cache_dir)
    for i, example in enumerate(ds):
        if i >= count:
            break
        yield example


def _parse_codecs(spec: str | None) -> List[Tuple[str, str, str]]:
    if not spec:
        return DEFAULT_CODECS
    parsed = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) == 1:
            key = parts[0]
            parsed.append((key, key, key))
        elif len(parts) == 2:
            key, registry_name = parts
            parsed.append((key, key, registry_name))
        elif len(parts) == 3:
            key, display_name, registry_name = parts
            parsed.append((key, display_name, registry_name))
        else:
            raise ValueError(f"Invalid codec spec: {item}")
    return parsed


def _load_codecs(codecs: List[Tuple[str, str, str]], strict: bool):
    available = set(codec_registry.list_codec())
    loaded = []
    missing = []
    for key, display_name, registry_name in codecs:
        if registry_name not in available:
            missing.append((display_name, registry_name))
            continue
        print(f"[codec] loading {display_name} ({registry_name})")
        loaded.append((key, display_name, registry_name, codec_registry.load_codec(registry_name)))

    if missing:
        msg = "\n".join(f"  - {name}: {registry}" for name, registry in missing)
        print("[warn] Missing SoundCodec registry entries; skipped:\n" + msg)
        print("[hint] Install/update Codec-SUPERB if you need llmcodec, auv, or bigcodec.")
        if strict:
            raise SystemExit(2)
    return loaded


def _synth_one(model, data_item: dict) -> Tuple[np.ndarray, int]:
    result = model.synth(data_item, local_save=False)
    if not isinstance(result, dict) or "audio" not in result:
        raise RuntimeError("model.synth returned an unexpected result")
    audio = result["audio"]
    wav = _mono_float32(audio["array"])
    sample_rate = int(audio.get("sampling_rate") or data_item["audio"]["sampling_rate"])
    return wav, sample_rate


def synthesize(args: argparse.Namespace) -> None:
    codecs = _parse_codecs(args.codecs)
    if args.list_codecs:
        print("\n".join(codec_registry.list_codec()))
        return

    if args.dry_run:
        print("[dry-run] domains:", ", ".join(DOMAIN_CONFIG))
        print("[dry-run] requested codecs:")
        for key, display_name, registry_name in codecs:
            print(f"  - {key}: {display_name} ({registry_name})")
        print("[dry-run] available SoundCodec entries:")
        for item in codec_registry.list_codec():
            print(f"  - {item}")
        return

    loaded_codecs = _load_codecs(codecs, strict=args.strict_codecs)
    out_dir = Path(args.out_dir)
    metadata: Dict[str, List[dict]] = {}

    for domain_key, cfg in DOMAIN_CONFIG.items():
        split = cfg["split"]
        prefix = cfg["prefix"]
        subdir = out_dir / cfg["out_subdir"]
        metadata[domain_key] = []

        print(f"[data] loading {args.dataset} split={split}")
        examples = _iter_examples(
            dataset_name=args.dataset,
            split=split,
            count=args.samples_per_domain,
            streaming=args.streaming,
            cache_dir=args.cache_dir,
        )

        for sample_idx, example in enumerate(examples, start=1):
            wav, sample_rate = _audio_from_example(example, seconds=args.seconds)
            stem = f"{prefix}{sample_idx}"
            gt_path = subdir / f"{stem}_gt.wav"
            _save_wav(gt_path, wav, sample_rate)

            source = str(example.get("source", split))
            item_meta = {
                "sample": stem,
                "split": split,
                "source": source,
                "ground_truth": gt_path.as_posix(),
                "reconstructions": {},
            }
            print(f"[sample] {domain_key}/{stem} source={source} sr={sample_rate}")

            data_item = {
                "id": f"{domain_key}-{sample_idx}",
                "audio": {"array": wav, "sampling_rate": sample_rate},
            }

            for key, display_name, registry_name, model in loaded_codecs:
                out_path = subdir / f"{stem}_{key}.wav"
                try:
                    rec, rec_sr = _synth_one(model, data_item)
                    _save_wav(out_path, rec, rec_sr)
                    item_meta["reconstructions"][key] = out_path.as_posix()
                    print(f"  [ok] {display_name} -> {out_path}")
                except Exception as exc:
                    print(f"  [fail] {display_name}: {exc}")
                    if args.strict_synthesis:
                        raise

            metadata[domain_key].append(item_meta)

    metadata_path = out_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[done] wrote metadata to {metadata_path}")
    if args.fast_exit:
        # Some Codec-SUPERB dependencies can crash during Python interpreter
        # finalization after all files have been written. Exit directly by
        # default for reliable batch/script usage.
        import sys
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="voidful/codec-superb-tiny")
    parser.add_argument("--out-dir", default="webpage/audio")
    parser.add_argument("--samples-per-domain", type=int, default=5)
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--streaming", action="store_true", default=True)
    parser.add_argument("--no-streaming", dest="streaming", action="store_false")
    parser.add_argument("--codecs", default=None,
                        help=("Comma list of key:display:registry entries. "
                              "Default uses llmcodec, auv, bigcodec, unicodec_24k, wavtokenizer."))
    parser.add_argument("--strict-codecs", action="store_true")
    parser.add_argument("--strict-synthesis", action="store_true")
    parser.add_argument("--fast-exit", action="store_true", default=True)
    parser.add_argument("--no-fast-exit", dest="fast_exit", action="store_false")
    parser.add_argument("--list-codecs", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    synthesize(parser.parse_args())


if __name__ == "__main__":
    main()

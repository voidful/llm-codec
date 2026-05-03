# LLM-Codec: Neural Audio Codec Meets Language Model Objectives

LLM-Codec is a training framework for adapting a neural audio codec so its
discrete speech tokens remain reconstructable while becoming easier for an
autoregressive language model to predict.

Standard neural codecs are optimized for waveform reconstruction. Spoken
language models, however, consume codec tokens with a next-token objective. This
objective mismatch can make acoustically valid token variations look like noise
to the LM. LLM-Codec addresses the mismatch by adding LM-facing training losses
to codec fine-tuning while keeping the deployed codec architecture unchanged.

This repository currently contains the codec training and checkpoint export
pipeline. Downstream TTS and SALMon evaluation scripts are not included in this
trimmed repo.

Links:

- Paper: https://arxiv.org/abs/2604.17852
- GitHub: https://github.com/voidful/llm-codec
- Hugging Face model: https://huggingface.co/voidful/llm-codec
- Project page: https://voidful.github.io/llm-codec/

## Highlights

- Future Token Prediction (FTP): Medusa-style heads predict multiple future
  audio tokens from frozen-LLM hidden states.
- Semantic Alignment (SA): audio-induced LLM representations are aligned with
  paired text representations using cosine and contrastive losses.
- Differentiable Gumbel bridge: hard Gumbel-Softmax keeps discrete forward
  tokens while allowing gradients to flow back to the codec encoder.
- Reconstruction-preserving codec training: mel, multi-scale mel,
  multi-resolution STFT, complex STFT, VQ, MPD/MSD GAN, and feature matching
  losses keep waveform quality stable.
- Inference cost is unchanged: auxiliary training heads are not required by the
  codec at deployment time.

## Paper Summary

The paper argues that reconstruction-trained codec tokens contain fine acoustic
variation that is useful for waveform fidelity but difficult for LMs to model.
LLM-Codec adds two regularizers during codec training:

1. FTP encourages local and multi-step token predictability.
2. SA keeps predictable tokens semantically grounded by aligning speech and text
   hidden states inside a frozen LLM.

Training uses AUV as the base codec and Qwen3-4B-Instruct as the frozen LM
backbone. The codec operates at 50 Hz with 20,480 audio tokens under the
canonical `<CODEC_*>` prefix.

## Main Results From The Paper

### Token Learnability

SALMon speech coherence accuracy after training a token-level speech LM:

| Tokenizer | Overall accuracy |
| --- | ---: |
| WavTok-L | 48.3 |
| BigCodec | 49.4 |
| UniCodec | 50.1 |
| AUV | 49.4 |
| LLM-Codec | 61.6 |

Token-level perplexity on LibriSpeech after 3 epochs of LM training:

| Tokenizer | Eval loss | Perplexity |
| --- | ---: | ---: |
| WavTok-L | 11.91 | 148,122 |
| UniCodec | 11.92 | 150,197 |
| BigCodec | 11.96 | 156,448 |
| AUV | 11.98 | 159,768 |
| LLM-Codec | 8.44 | 4,617 |

### Reconstruction Quality

Codec-SUPERB-tiny speech reconstruction:

| Model | Mel lower is better | STFT lower is better | PESQ higher is better | STOI higher is better |
| --- | ---: | ---: | ---: | ---: |
| AUV base | 0.762 | 1.648 | 2.094 | 0.850 |
| LLM-Codec | 0.724 | 1.599 | 2.102 | 0.859 |

The reported speech Mel distance improves by 5.0 percent over AUV while token
perplexity drops by about 35x.

## Repository Structure

```text
.
|-- train.py              # Main LLM-Codec training loop
|-- run_codec_train.sh    # 25k-step training recipe used by this repo
|-- extract_upload.py     # Export checkpoint artifacts and push to HF Hub
|-- llm_codec/
|   |-- codec.py          # AUV codec wrapper with differentiable encode/decode
|   |-- system.py         # Unified AUV + Qwen system wrapper
|   |-- qwen_ar.py        # Qwen tokenizer/audio-token adapter
|   |-- gan.py            # MPD/MSD discriminators and GAN losses
|   |-- losses.py         # Mel, STFT, complex STFT, and multi-scale losses
|   |-- schedules.py      # Warmup, ramp, and cosine schedules
|   |-- utils.py          # Audio I/O, stats, and visualization helpers
|   `-- wb.py             # Optional Weights & Biases wrapper
`-- webpage/index.html    # Project webpage asset
```

## Installation

Use a CUDA environment with enough memory to host AUV, Qwen3-4B-Instruct, and
the audio discriminators.

```bash
pip install git+https://github.com/voidful/AUV.git
pip install torch torchaudio transformers datasets huggingface_hub wandb numpy psutil
```

If you plan to push artifacts to Hugging Face Hub:

```bash
huggingface-cli login
```

## Data And Checkpoints

`train.py` loads LibriSpeech directly from Hugging Face Datasets:

- training split: `librispeech_asr`, config `clean`, split `train.100`
- validation split: `librispeech_asr`, config `clean`, split `validation`

The default script expects a base AUV checkpoint at:

```text
./auv.pt
```

Use `--auv_ckpt` to point to a different checkpoint and `--cache_dir` to control
the Hugging Face dataset cache.

## Webpage Audio Examples

The project page reads comparison clips from `webpage/audio/`. Use the latest
Codec-SUPERB `SoundCodec` interface to synthesize Codec-SUPERB-tiny examples:

```bash
git clone --depth 1 https://github.com/voidful/Codec-SUPERB.git /tmp/Codec-SUPERB
CODEC_SUPERB_ROOT=/tmp/Codec-SUPERB \
python scripts/synthesize_webpage_audio.py
```

The default command samples the `Speech`, `Music`, and `Audio` splits from
`voidful/codec-superb-tiny`, taking five examples per domain. It writes
ground-truth and reconstructed wavs for the paper baseline set:

- `llmcodec`
- `auv`
- `bigcodec_1k`
- `unicodec_24k`
- `wavtokenizer_24k_large_600_4096`

This repository includes the generated webpage set: 3 domains x 5 examples x
6 audio tracks (ground truth plus five codecs), for 90 wav files total.

## Training

Run the repo recipe:

```bash
bash run_codec_train.sh
```

The script trains for 25k steps and writes outputs to `runs/llm_codec`.

Important defaults:

| Setting | Value |
| --- | --- |
| Base codec | AUV |
| LLM backbone | `Qwen/Qwen3-4B-Instruct-2507` |
| Audio token prefix | `<CODEC_` |
| Audio vocabulary size | 20,480 |
| Segment length | 4 seconds |
| Batch / grad accumulation | 1 / 10 |
| Codec optimizer | SGD, momentum 0.9 |
| Encoder LR | `5e-6` |
| Decoder LR | `5e-6` |
| Audio embedding and Medusa LR | `1e-4` |
| Max steps | 25,000 |
| FTP | lambda 0.2, K 5, delay 10k, ramp 2k |
| SA | cosine 0.1, contrastive 0.05, delay 12k, warmup 2k |
| Gumbel bridge | tau 1.0 to 0.3 over 20k steps |
| GAN | MPD/MSD hinge GAN, feature matching, R1 every 16 steps |

Training schedule:

```text
step      0          10k        12k        14k                  25k
GAN-D     active throughout
GAN-G     off        active
FTP       off        ramp       active
SA        off                   ramp       active
```

The total loss combines:

```text
L_total =
  L_mel + L_ms_mel + L_mr_stft + L_cstft + L_vq
  + L_bridge + L_FTP + L_SA_cosine + L_SA_contrast
  + L_GAN + L_feature_matching
```

### Useful Manual Overrides

Use a different output directory:

```bash
python train.py --auv_ckpt ./auv.pt --out_dir runs/my_llm_codec
```

Resume from a checkpoint:

```bash
python train.py \
  --auv_ckpt ./auv.pt \
  --resume runs/llm_codec/ckpt_step10000.pt \
  --out_dir runs/llm_codec
```

Disable Weights & Biases logging:

```bash
python train.py \
  --auv_ckpt ./auv.pt \
  --wandb_project "" \
  --out_dir runs/no_wandb
```

## Training Outputs

The training directory contains:

```text
runs/llm_codec/
|-- tokenizer/                  # Qwen tokenizer extended with <CODEC_*> tokens
|-- ckpt_step*.pt               # Periodic checkpoints
|-- ckpt_final_step*.pt         # Final checkpoint
`-- val_step*_sample*_codec.wav # Validation reconstructions
```

Checkpoint contents include:

- AUV codec weights under `auv`
- Qwen audio-token embeddings under `qwen_embed`
- Qwen LM head weights under `qwen_lm_head`
- tokenizer path under `tok_dir`
- optimizer states for resume
- Medusa head state dicts for analysis or continued training

## Export And Upload

Export a trained checkpoint and push model artifacts to Hugging Face Hub:

```bash
python extract_upload.py \
  --ckpt runs/llm_codec/ckpt_final_step24999.pt \
  --repo_id your-name/llm-codec \
  --qwen_model Qwen/Qwen3-4B-Instruct-2507
```

The export script:

- saves codec weights as `llm-codec.pt`
- loads the base Qwen model
- applies the trained audio-token embeddings
- migrates legacy `<AUV_*>` or `<SPEECH_*>` tokens to canonical `<CODEC_*>`
- pushes tokenizer, Qwen model artifacts, and codec weights to the target repo

## Implementation Notes

- `AUVCodecWrapper` keeps AUV encode/decode differentiable and aligns decoder
  output length to the input waveform.
- `QwenAR` extends the tokenizer with `<CODEC_0>` to `<CODEC_20479>` and freezes
  the LLM except for input embeddings.
- `GumbelBridge` is implemented inside `train.py` and projects codec latents to
  audio-token logits.
- FTP heads are initialized from the frozen Qwen LM head rows corresponding to
  audio-token IDs.
- SA aligns middle-to-high Qwen layers, from `L/3` to `0.8L`, using last-token
  pooled audio and text hidden states.
- The SA memory bank stores recent text representations and uses label-smoothed
  contrastive cross entropy.
- BatchNorm layers inside the codec are kept in eval mode during training to
  avoid drifting pretrained statistics.

## Current Scope

This repo is intentionally focused on training and exporting the LLM-aligned
codec. The paper reports downstream SALMon, token-LM perplexity, and
Codec-SUPERB-tiny results, but the corresponding evaluation pipelines are not
part of the current repository snapshot.

## Citation

```bibtex
@article{chung2026llm,
  title={LLM-Codec: Neural Audio Codec Meets Language Model Objectives},
  author={Chung, Ho-Lam and Chen, Yiming and Lee, Hung-yi},
  journal={arXiv preprint arXiv:2604.17852},
  year={2026}
}
```

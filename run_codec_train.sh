#!/bin/bash
# run_codec_train.sh
# Train the LLM-Codec (audio1dcodec) with staggered LLM → Semantic schedule.
#
# Schedule overview (25k steps):
#
#   Step    0        10k     12k     14k                  25k
#           │         │       │       │                    │
#   GAN-D   ████████████████████████████████████████████████  (all steps)
#   GAN-G   ─────────████████████████████████████████████████  (10k+, λ=1.0)
#   FM      ─────────████████████████████████████████████████  (10k+, λ=1.5)
#   LLM     ──────────╱╲█████████████████████████████████████  (10k→12k ramp, 12k+ full λ=0.2)
#   Semantic────────────────╱╲███████████████████████████████  (12k→14k ramp, 14k+ full)
#
#   ╱╲ = linear warmup ramp    █ = full weight active
#
# Loss weights (unchanged):
#   lambda_ftp  = 0.2
#   lambda_sa_cosine  = 0.1
#   lambda_sa_contrast= 0.05
#
# Schedule changes (vs previous):
#   ftp_delay_steps:  8000  → 10000  (sync with GAN-G start at d_only_steps)
#   sa_delay_steps:  12000 → 12000  (2k after LLM starts, staggered)
#   ftp_ramp:         2000  → 2000   (LLM full at step 12k)
#   sa_warmup:       2000  → 2000   (SA full at step 14k)

set -e

python train.py \
    --auv_ckpt ./auv.pt \
    --out_dir runs/llm_codec \
    --wandb_project llm-codec \
    --wandb_run llm_codec \
    --mp_dtype bf16 \
    --batch_size 1 \
    --grad_accum_steps 10 \
    --num_workers 8 \
    --max_steps 25000 \
    \
    --mel_rms_norm \
    --lambda_mel 1.5 \
    --enable_ms_mel \
    --lambda_ms_mel 0.5 \
    --enable_mr_stft \
    --lambda_mr_stft 0.5 \
    --enable_cstft \
    --lambda_cstft 0.8 \
    --cstft_phase_weight 0.5 \
    --lambda_vq 1.0 \
    --gan_pause_steps 0 \
    --fm_pause_share 0.99 \
    \
    --enable_gan \
    --lambda_gan 1.0 \
    --gan_loss_type hinge \
    --lr_d 1e-4 \
    --d_update_every 1 \
    --r1_gamma 2.0 \
    --d_reg_every 16 \
    --d_only_steps 10000 \
    --gan_warmup 0 \
    --gan_ramp 0 \
    \
    --enable_phase_jitter \
    --phase_jitter_max 24 \
    \
    --opt_codec sgd \
    --lr_enc 5e-6 \
    --lr_dec 5e-6 \
    --grad_clip 15.0 \
    \
    --lambda_sa_cosine 0.1 \
    --lambda_sa_contrast 0.05 \
    --sa_delay_steps 12000 \
    --sa_warmup 2000 \
    --sa_logit_scale 5.0 \
    \
    --lambda_ftp 0.2 \
    --ftp_delay_steps 10000 \
    --ftp_ramp 2000 \
    --lr_embed 1e-4 \
    \
    --lambda_fm_init 1.5 \
    --lambda_fm_final 1.0 \
    --val_every 2000 \
    --save_every 5000 \
    --gan_amp fp32 \
    --nan_guard

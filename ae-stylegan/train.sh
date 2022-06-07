#!/bin/sh

python train.py \
--path ../celeba/train_yo \
--which_latent w_plus \
--lambda_rec_w 0 \
--iter 200000 \
--size 128 \
--name ffhq_aegan_wplus_decoupled \
--log_every 500 \
--save_every 2000 \
--eval_every 2000 \
--dataset imagefolder \
--inception inception_ffhq128.pkl \
--n_sample_fid 10000 \
--decouple_d \
--lambda_rec_d 0 \
--g_reg_every 0 \
--batch 1 \
--lr 0.0025 \
--r1 0.2048 \
--ema_kimg 5 \
--disc_iter_start 30000 \
--which_metric  \
# --use_adaptive_weight \
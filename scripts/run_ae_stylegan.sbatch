#!/bin/bash

#SBATCH --job-name=AE-StyleGAN
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s286886@studenti.polito.it
#SBATCH --workdir=/home/mla_group_13/FaceFilter/ae-stylegan
#SBATCH --partition=cuda
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/mla_group_13/experiments_output/ae-stylegan_%j.log
#SBATCH --mem=8G
#SBATCH --gres=gpu:1

module load nvidia/cudasdk/11.6
module load intel/python/3/2019.4.088

source /home/mla_group_13/.bashrc
conda activate env-stylegan

python train.py \
--path /home/mla_group_13/FaceFilter/celeba/train_ab \
--which_latent w_plus \
--log_root /home/mla_group_13/experiments_output/ae-stylegan_$SLURM_JOB_ID/ \
--lambda_rec_w 0 \
--iter 2500 \
--size 128 \
--name ffhq_aegan_wplus_joint \
--log_every 100 \
--save_every 1000 \
--eval_every 500 \
--dataset imagefolder \
--inception inception_ffhq128.pkl \
--n_sample_fid 10000 \
--lambda_rec_d 0.25 \
--lambda_fake_d 0.9 \
--lambda_fake_g 0.9 \
--g_reg_every 0 \
--batch 16 \
--lr 0.0025 \
--r1 0.2048 \
--ema_kimg 5 \
--disc_iter_start 30000 \
# --which_metric fid_sample fid_recon


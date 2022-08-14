#!/bin/bash

#SBATCH --job-name=AE-StyleGAN-E-10
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s287598@studenti.polito.it
#SBATCH --workdir=/home/mla_group_13/FaceFilter/ae-stylegan
#SBATCH --partition=cuda
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/mla_group_13/experiments_output/ae-stylegan_%j.log
#SBATCH --mem=8G
#SBATCH --gpus-per-node=1

module load nvidia/cudasdk/11.6
module load intel/python/3/2019.4.088

source /home/mla_group_13/.bashrc
conda activate env-stylegan

device="cuda"
if [ -z $CUDA_VISIBLE_DEVICES ]; then
    device="cpu"
fi

n_step_e=10

lambda_rec_w=0
lambda_rec_d=0.25
lambda_fake_d=0.9
lambda_fake_g=0.9

r1=0.2048

echo AE-StyleGAN n_step_e=${n_step_e} lambda_rec_w=${lambda_rec_w} lambda_rec_d=${lambda_rec_d} lambda_fake_d=${lambda_fake_d} lambda_fake_g=${lambda_fake_g} r1=${r1}

python train.py \
--path /home/mla_group_13/FaceFilter/celeba/train_ab \
--which_latent w_plus \
--log_root /home/mla_group_13/experiments_output/ae-stylegan_$SLURM_JOB_ID/ \
--resume \
--n_step_e $n_step_e \
--lambda_rec_w $lambda_rec_w \
--iter 100000 \
--size 128 \
--name ffhq_aegan_wplus_joint \
--log_every 100 \
--save_every 1000 \
--eval_every 50 \
--dataset imagefolder \
--inception inception_celeba.pkl \
--n_sample_fid 10000 \
--lambda_rec_d $lambda_rec_d \
--lambda_fake_d $lambda_fake_d \
--lambda_fake_g $lambda_fake_g \
--g_reg_every 0 \
--batch 16 \
--lr 0.0025 \
--r1 $r1 \
--ema_kimg 5 \
--disc_iter_start 30000 \
--device $device \
--which_metric fid_sample fid_recon \
--ckpt /home/mla_group_13/experiments_output/ae-stylegan_265711/ffhq_aegan_wplus_joint/weight/011000.pt \
# --which_metric fid_sample fid_recon
# --inception inception_ffhq128.pkl


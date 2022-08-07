#!/bin/bash

#SBATCH --job-name=Cycle-StyleGAN
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s286886@studenti.polito.it
#SBATCH --workdir=/home/mla_group_13/FaceFilter/ae-stylegan
#SBATCH --partition=cuda
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/mla_group_13/experiments_output/cycle-stylegan_%j.log
#SBATCH --mem=8G
#SBATCH --gres=gpu:1

module load nvidia/cudasdk/11.6
module load intel/python/3/2019.4.088

source /home/mla_group_13/.bashrc
conda activate env-stylegan

if [ ! -d $TMPDIR/data/young ]; then
    mkdir -p $TMPDIR/data/young
    tar xf $HOME/train_young.tar.gz -C $TMPDIR/data/young
fi

if [ ! -d $TMPDIR/data/old ]; then
    mkdir -p $TMPDIR/data/old
    tar xf $HOME/train_old.tar.gz -C $TMPDIR/data/old
fi

if [ ! -d $TMPDIR/data-test/young ]; then
    mkdir -p $TMPDIR/data-test/young
    tar xf $HOME/test_young.tar.gz -C $TMPDIR/data-test/young
fi

if [ ! -d $TMPDIR/data-test/old ]; then
    mkdir -p $TMPDIR/data-test/old
    tar xf $HOME/test_old.tar.gz -C $TMPDIR/data-test/old
fi

python train-cycle-ema.py \
--path $TMPDIR/data \
--path-test $TMPDIR/data-test \
--which_latent w_plus \
--log_root /home/mla_group_13/experiments_output/cycle-stylegan_${SLURM_JOB_ID}/ \
--resume \
--ckpt /home/mla_group_13/experiments_output/ae-stylegan_266738/ffhq_aegan_wplus_joint/weight/075000.pt \
--iter 500 \
--size 128 \
--name ffhq_aegan_wplus_joint \
--log_every 1 \
--save_every 200 \
--eval_every 25 \
--dataset imagefolder \
--inception inception_ffhq128.pkl \
--n_sample_fid 10000 \
--batch 32 \
--lr 0.0001 \
--r1 0.2048 \
--ema_kimg 5 \
--disc_iter_start 30000 \
--n_sample 64 \
--lambda_vgg 0 \
#--device cpu
#--lambda_pix 10
# --device cpu
# --which_metric fid_sample fid_recon


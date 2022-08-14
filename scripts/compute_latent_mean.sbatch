#!/bin/bash

#SBATCH --job-name=latent_mean
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s88423@studenti.polito.it
#SBATCH --workdir=/home/mla_group_13/FaceFilter/ae-stylegan
#SBATCH --partition=cuda
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/mla_group_13/experiments_output/latent_mean_%j.log
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1

module load nvidia/cudasdk/11.6
module load intel/python/3/2019.4.088

source /home/mla_group_13/.bashrc
conda activate ae-style-mean

python compute_latent_mean.py \
    --data-path /home/mla_group_13/FaceFilter/celeba/train_old-young \
    --model-path /home/mla_group_13/experiments_output/ae-stylegan_266738/ffhq_aegan_wplus_joint/weight/075000.pt \
    --out /home/mla_group_13/experiments_output/latent_mean/mean.pt \
    --batch-size 4

#!/bin/bash

#SBATCH --job-name=CycleGAN_50
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s287598@studenti.polito.it
#SBATCH --workdir=/home/mla_group_13/FaceFilter/FaceAging-by-cycleGAN
#SBATCH --partition=cuda
#SBATCH --time=100:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/mla_group_13/experiments_output/cycle_gan_%j.log
#SBATCH --mem=8G
#SBATCH --gpus-per-node=1

module load nvidia/cudasdk/10.1
module load intel/python/3/2019.4.088

source /home/mla_group_13/.bashrc
conda activate mla39

PORT=8103
visdom -port $PORT &

EPOCH=50
REM_EPOCH=$((200-$EPOCH))
BALANCED="_BALANCED"
mkdir ./checkpoints/9_wiki_fine_tune_male/"$EPOCH"_epochs"$BALANCED" 

echo $CUDA_VISIBLE_DEVICES

python train.py --dataroot ../celeba \
    --name 9_wiki_fine_tune_male \
    --model cycle_gan --gpu_ids 0 \
    --display_freq 5 --print_freq 5 \
    --use_pretrained_model \
    --pretrained_model_name 9_wiki_fine_tune_male \
    --pretrained_model_subname G_A,G_B,D_A,D_B \
    --pretrained_model_epoch 200 \
    --epoch_count $REM_EPOCH \
    --display_server localhost \
    --batch_size 8 \
    --save_epoch_freq 200 \
    --display_port $PORT

mv ./checkpoints/9_wiki_fine_tune_male/end* ./checkpoints/9_wiki_fine_tune_male/"$EPOCH"_epochs"$BALANCED"

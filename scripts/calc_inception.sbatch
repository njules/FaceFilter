#!/bin/bash

#SBATCH --job-name=Cycle-StyleGAN
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s286886@studenti.polito.it
#SBATCH --workdir=/home/mla_group_13/FaceFilter/ae-stylegan
#SBATCH --partition=cuda
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/mla_group_13/experiments_output/inception_%j.log
#SBATCH --mem=8G
#SBATCH --gpus-per-node=1

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

#python calc_inception.py --size 128 --batch 64 --name celeba --dataset imagefolder --path $TMPDIR/data
python calc_inception.py --size 128 --batch 64 --name celeba --dataset imagefolder --path $TMPDIR/data --by-class young
python calc_inception.py --size 128 --batch 64 --name celeba --dataset imagefolder --path $TMPDIR/data --by-class old


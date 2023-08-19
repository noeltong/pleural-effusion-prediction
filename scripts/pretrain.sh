#!/bin/bash

#SBATCH --job-name=pretrain
#SBATCH --partition=normal
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --mail-type=all
#SBATCH --mail-user=tongshq@shanghaitech.edu.cn
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=10-00:00:00
#SBATCH --nodelist=sist_gpu49

torchrun --nproc_per_node=4 main.py --mode train --config /public/home/tongshq/pleural-effusion-prediction/configs/pretrain.py --workdir pertrain_vit_b
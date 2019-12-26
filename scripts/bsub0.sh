#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -J myJob
#BSUB -n 1
#BSUB -W 10:00
#BSUB -R "rusage[mem=12GB]"
#BSUB -o ../logs/conv%J.out

module load python3/3.6.2
module load cuda/8.0
module load cudnn/v7.0-prod-cuda8

echo "Running script..."
python3 train.py 10000 CONV 0.01 emoving

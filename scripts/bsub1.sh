#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -J myJob
#BSUB -n 1
#BSUB -W 20:00
#BSUB -R "rusage[mem=20GB]"
#BSUB -o ../logs/part%J.out

module load python3/3.6.2
module load cuda/8.0
module load cudnn/v7.0-prod-cuda8

echo "Running script..."
python3 train.py 5000 PARTICLECONV 0.01 emoving

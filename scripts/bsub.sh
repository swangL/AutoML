#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -J myJob
#BSUB -n 1
#BSUB -W 10:00
#BSUB -R "rusage[mem=6GB]"
#BSUB -o ../logs/gruco%J.out

module load python3/3.6.2
module load cuda/8.0
module load cudnn/v7.0-prod-cuda8

echo "Running script..."
<<<<<<< HEAD
python3 train.py 10000 MOONS 0.01 gct
# python3 train.py 10000 MNIST 0.01
=======
python3 train.py 10000 MOONS 0.01 egcct
python3 train.py 10000 MNIST 0.01 egcct
>>>>>>> 9c238499891353a6a2095fb14b9f400090c0ef4f

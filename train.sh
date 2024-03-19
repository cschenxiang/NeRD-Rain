#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 7-00:00:00

#conda activate pytorch

python train.py | tee logs_Rain200L.txt
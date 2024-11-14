#!/bin/bash
#SBATCH --job-name=t5_xxl_train
#SBATCH --gres=gpu:a100-80:1
#SBATCH --time=100:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=<email>

srun python t5_train.py
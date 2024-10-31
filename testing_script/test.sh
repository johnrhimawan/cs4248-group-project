#!/bin/bash

#SBATCH --job-name=gpujob
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=e0550468@u.nus.edu

srun python3 test-t5.py

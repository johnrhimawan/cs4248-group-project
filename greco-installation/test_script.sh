#!/bin/bash

#SBATCH --job-name=test-greco
#SBATCH --gres=gpu:nv:2
#SBATCH --time=24:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=e0851472@u.nus.edu

python3 test.py 

#!/bin/bash

#SBATCH --job-name=eval-conll14
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=e0550468@u.nus.edu

srun python3 ../m2scorer/scripts/m2scorer.py result/t5-conll14-2.txt ../data/test/conll14st-test-data/alt/official-2014.combined-withalt.m2

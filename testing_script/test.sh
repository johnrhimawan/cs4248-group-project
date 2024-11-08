#!/bin/bash

#SBATCH --job-name=testing
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=e0550468@u.nus.edu

pipenv shell
#srun python3 testing-code/test-t5.py --TEST_SET CONLL14
srun python3 testing-code/test-t5.py --TEST_SET BEA2019

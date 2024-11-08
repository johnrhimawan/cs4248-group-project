#!/bin/bash

#SBATCH --job-name=install_packages
#SBATCH --time=100:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=e0550468@u.nus.edu

pipenv shell
pipenv install torch torchvision torchaudio transformers huggingface_hub accelerate spacy datasets peft
srun python3 -m spacy download en_core_web_sm

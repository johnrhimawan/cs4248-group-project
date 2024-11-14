#!/bin/bash
#SBATCH --job-name=llama_finetuning
#SBATCH --gres=gpu:h100-47:1
#SBATCH --time=100:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=<email>

srun python finetune_llama_script.py
# Fine-tuning Flan-T5

To fine-tune Flan-T5, you can simply execute the bash script on the SoC compute cluster which will run on NVIDIA A100 GPU.

`sbatch finetune_t5.sh`

Feel free to change the settings as you prefer.

```bash
#!/bin/bash
#SBATCH --job-name=t5_xxl_train
#SBATCH --gres=gpu:a100-80:1
#SBATCH --time=100:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=<email>

srun python t5_train.py
```

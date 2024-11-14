# Fine-tuning Llama 3.1

To fine-tune Llama 3.1, you can simply execute the bash script on the SoC compute cluster which will run on NVIDIA H100 GPU.

`sbatch finetune_llama.sh`

Feel free to change the settings as you prefer.

```bash
#!/bin/bash
#SBATCH --job-name=llama_finetuning
#SBATCH --gres=gpu:h100-47:1
#SBATCH --time=100:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=<email>

srun python finetune_llama_script.py
```

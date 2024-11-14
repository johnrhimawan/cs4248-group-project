# CS4248 Group Project

**Group Name: G08**

**Team Members:**

- Christopher Adrian
- Ethan Chen Ee Shuen
- Paul David Christopher Golling
- John Russell Himawan
- Nguyen Viet Anh

## Installation

Run `sbatch install.sh`.

## Fine-tuning

To fine-tune the models, see [llama](./baseline_llama/) or [T5](./baseline_t5/).

## Evaluating Models

See [testing_script](./testing_script/).

## Ensembling with GRECO

See [greco-ensemble](./greco-ensemble/).

## Appendix: How to Use SoC Compute Cluster

SoC has published documentation on how to [use the compute cluster](https://dochub.comp.nus.edu.sg/cf/services/compute-cluster) and [a quickstart for using Slurm](https://dochub.comp.nus.edu.sg/cf/guides/compute-cluster/slurm-quick).

You need to use Slurm to submit jobs to the compute cluster to use the GPUs. The following is a simple example of how to submit a job to the compute cluster:

1. Create a batch script, e.g. `somejob.sh`:

```bash
#!/bin/bash

#SBATCH --job-name=gpujob
#SBATCH --gpus=1
#SBATCH --time=5:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=<your-email>

srun python somegpuprogram.py
```

The above script will request 1 GPU for 5 hours and send you an email when the job starts, ends, or fails. You can edit the script with the functions that suit your needs.

2. Submit a batch job, e.g. `sbatch somejob.sh`.

3. Now just wait for the job to finish. You can check the status of your job with `squeue -u <username>` or list job accounting information with `sacct`. If you need to cancel a job, you can do so with `scancel <jobid>`.

4. Useful note: You probably want to utilize a GPU in the SoC Compute Cluster. There are a few different partitions available: normal, gpu, long, gpu-long. For training models, you should use gpu-long. More information [here](https://dochub.comp.nus.edu.sg/cf/guides/compute-cluster/gpu).

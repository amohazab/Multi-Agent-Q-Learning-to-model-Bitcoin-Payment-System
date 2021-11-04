#!/bin/bash
#SBATCH --time=60:00:00
#SBATCH --mem=5G
#SBATCH --job-name=epsilon_greedy_2
#SBATCH --output=res_%a.out
#SBATCH --array=0-1000

srun python eps_greed_github.py $SLURM_ARRAY_TASK_ID > RESULT_$SLURM_ARRAY_TASK_ID.json
#!/bin/bash
#SBATCH --time=3-00:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=10GB

module load Python/3.10.4-GCCcore-11.3.0
module load CUDA/12.4.0

source $HOME/venvs/pet-ct_env/bin/activate

python ./main.py "$@"

# Run as: sbatch --job-name="job_name_here" training_full_job.sh --type CT_only or PET_CTcat
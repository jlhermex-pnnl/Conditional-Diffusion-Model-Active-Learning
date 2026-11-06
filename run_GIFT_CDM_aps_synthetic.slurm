#!/bin/bash
#SBATCH -A dvztj
#SBATCH --job-name=CDM_GIFT_UncertaintySampler
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=a100_shared
#SBATCH --time=4-00:00:00
#SBATCH --output=CDM_GIFT_UncertaintySampler_%j.out
#SBATCH --error=CDM_GIFT_UncertaintySampler_%j.err

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Load conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ttnn

# Clear Python cache to ensure updated code is used
rm -rf __pycache__
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null

# Run the script
srun python -u CDM_Simple_WSSE_explore_e2_200_resume.py

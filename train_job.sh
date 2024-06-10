#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --nodes=2
#SBATCH --mem=16GB
#SBATCH --ntasks-per-node=1
#SBATCH --output=job-newblockedcv-%j.log
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1

module load PyTorch
pip install -r requirements.txt
python3 main.py


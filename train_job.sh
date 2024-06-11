#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --mem=8GB
#SBATCH --ntasks-per-node=1
#SBATCH --output=training-%j.log
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1

module load PyTorch
pip install -r requirements.txt
python3 main.py


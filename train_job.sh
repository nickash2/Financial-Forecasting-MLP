#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --nodes=2
#SBATCH --mem=8000
#SBATCH --ntasks-per-node=2
#SBATCH --output=job-%j.log
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:2

module load PyTorch
pip install -r requirements.txt
python3 main.py



#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --nodes=2
#SBATCH --mem=16GB
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --mail-type=END

module load PyTorch
pip install -r requirements.txt
python3 main.py



#!/bin/bash
#SBATCH --mem=100g
#SBATCH --time=4:00:00
#SBATCH --output=log-%j.out
#SBATCH --error=log-%j.err

python -u knn_pbmc.py

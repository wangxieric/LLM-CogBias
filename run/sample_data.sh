#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=500G
#SBATCH --time=96:00:00

module load CUDA/12.4.0
export HF_HOME="/mnt/parscratch/users/ac1xwa/huggingface"
module load Anaconda3/2022.05
source activate cogbias
huggingface-cli login --token <token>
python ../src/sample_data.py
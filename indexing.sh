#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:4
#SBATCH --mem=400G
#SBATCH --time=96:00:00

export JAVA_HOME="/mnt/parscratch/users/ac1xwa/jdk-21.0.2"
export PATH=$JAVA_HOME/bin:$PATH
export HYDRA_FULL_ERROR=1
export HF_HOME="/mnt/parscratch/users/ac1xwa/huggingface"
module load Anaconda3/2022.05
source activate pythia
python indexing.py 
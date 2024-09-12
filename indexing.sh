#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time=96:00:00

export JAVA_HOME="/mnt/parscratch/users/ac1xwa/jdk-21.0.2"
export PATH=$JAVA_HOME/bin:$PATH
export HYDRA_FULL_ERROR=1
export HF_HOME="/mnt/parscratch/users/ac1xwa/huggingface"
/mnt/parscratch/users/ac1xwa/anaconda/.envs/pythia/bin/python indexing.py 
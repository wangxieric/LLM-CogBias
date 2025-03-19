#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=300G
#SBATCH --time=96:00:00

module load CUDA/12.4.0
export HF_HOME="/mnt/parscratch/users/ac1xwa/huggingface"
module load Anaconda3/2022.05
source activate cogbias
huggingface-cli login --token <token>
python src/eval_language_modelling.py --model_name "XiWangEric/literary-classicist-llama3" --dataset_name "mmlu" --task "multi_subject_mc"
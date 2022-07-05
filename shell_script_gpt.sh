#!/bin/bash
#SBATCH --gres=gpu:1
export PATH=/vol/bitbucket/ajl115/myvenv/bin/:$PATH
source activate
source /vol/cuda/11.0.3-cudnn8.0.5.39/setup.sh
TERM=vt100 # or TERM=xterm
python3 huggingface.py --train_data_file data/epzh-for-gpt.txt --output_dir gpt2-empatheticpersonas --model_type gpt2 --model_name_or_path sberbank-ai/mGPT
/usr/bin/nvidia-smi
uptime
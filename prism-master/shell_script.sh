#!/bin/bash
#SBATCH --gres=gpu:1
export PATH=/vol/bitbucket/ajl115/multilingual-psychotherapy-chatbot/prism-master/bin/:$PATH
source activate
source /vol/cuda/11.0.3-cudnn8.0.5.39/setup.sh
TERM=vt100 # or TERM=xterm
python3 PRISM-SRC_score.py
/usr/bin/nvidia-smi
uptime
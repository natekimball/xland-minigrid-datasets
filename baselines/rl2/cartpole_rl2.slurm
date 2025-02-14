#!/bin/bash

#SBATCH --job-name=cartpole
#SBATCH --output=./slurm_out/output_%j.txt
# #SBATCH --mail-type=begin,end,fail
# #SBATCH --mail-user=tma5gv@virginia.edu
#SBATCH --ntasks=1
# #SBATCH --cpus-per-task=38
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --constraint=a4000
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --partition=gpu

module purge
module load gcccore/13.3.0 python/3.12.3 cuda/12.3.0 cudnn/8.9.7.29-CUDA-12.3.0

if [ ! -d "ENV" ]; then
    python3 -m venv ENV
    source ENV/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    pip install flash-attn --no-build-isolation
else
    source ENV/bin/activate
fi

python cartpole_rl2.py
#!/bin/bash

#SBATCH --job-name=xland
#SBATCH --output=./slurm_out/output_%j.txt
# #SBATCH --mail-type=begin,end,fail
# #SBATCH --mail-user=tma5gv@virginia.edu
#SBATCH --ntasks=1
# #SBATCH --cpus-per-task=38
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --constraint=a100_80gb
#SBATCH --gres=gpu:1
#SBATCH --time=0-04:00:00
#SBATCH --partition=gpu


module load gcccore/13.3.0 python/3.12.3 cuda/12.4.0

nvidia-smi

BASELINE=ad
cd baselines/$BASELINE/

if [ ! -d "ENV" ]; then
    python3 -m venv ENV
    source ENV/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    pip install flash-attn --no-build-isolation
else
    source ENV/bin/activate
fi

export WORLD_SIZE=$(nvidia-smi -L | wc -l)
echo WORLD_SIZE=$WORLD_SIZE

DATA_PATH=/bigtemp/$USER/trivial-10k.hdf5

deepspeed --num_gpus "$WORLD_SIZE" train_xland.py \
    --config_path='configs/xland.yaml' \
    --learning_histories_path=$DATA_PATH
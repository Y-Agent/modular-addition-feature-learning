#!/bin/bash

#SBATCH --job-name=tk_module_addition_feature # Job name
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --qos=qos_zhuoran_yang
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --output=slurm_output/%j.out
#SBATCH --error=slurm_output/%j.err
#SBATCH --requeue

# Set working directory explicitly
WORK_DIR=/home/jh3439/modular-addition-feature-learning

echo '-------------------------------'
cd ${WORK_DIR}
echo "Working directory: $(pwd)"
echo Running on host $(hostname)
echo Time is $(date)
echo '-------------------------------'
echo -e '\n\n'

export PROCS=${SLURM_CPUS_ON_NODE}

module load CUDA
module load cuDNN
module load miniconda

# Initialize conda for bash - try multiple methods
source $(conda info --base)/etc/profile.d/conda.sh
conda activate llm_base

echo "Python path: $(which python)"
echo "Python version: $(python --version)"
echo "Conda environment: $CONDA_DEFAULT_ENV"

echo "Starting experiments..."
echo "============================================================="

cd src

# Use explicit Python path from llm_base environment
/gpfs/radev/home/jh3439/.conda/envs/llm_base/bin/python module_nn.py --init_type random --act_type ReLU --optimizer AdamW --init_scale 0.1
#python module_nn.py --init_type random --act_type ReLU --optimizer SGD --lr 0.1 --init_scale 0.01
#python module_nn.py --init_type single-freq --act_type Quad --optimizer SGD --lr 0.1 --init_scale 0.02
#python module_nn.py --init_type single-freq --act_type ReLU --optimizer SGD --lr 0.01 --init_scale 0.002
#python module_nn.py --init_type random --act_type Quad --optimizer SGD --lr 0.1 --init_scale 0.1


#python module_nn.py --init_type random --act_type ReLU --optimizer AdamW --init_scale 0.1 --frac_train 0.75 --weight_decay 2 --lr 1e-4 --num_epochs 50000 --d_mlp 128
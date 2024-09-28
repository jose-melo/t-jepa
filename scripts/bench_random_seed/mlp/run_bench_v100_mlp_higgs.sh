#!/bin/bash
#SBATCH --job-name=mlp_tuned_higgs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=12:00:00



echo "Checking python: $(python --version) $(which python)"

python -c "import torch; print('Torch is available: ', torch.cuda.is_available())"

DESIRED_CUDA_VERSION=12
NVCC_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]*\).\([0-9]*\),.*/\1.\2/')
NVIDIA_SMI_VERSION=$(nvidia-smi | grep "CUDA Version" | sed 's/.*CUDA Version: \([0-9]*\).\([0-9]*\).*/\1.\2/')

echo "NVCC CUDA Version: $NVCC_VERSION"
echo "NVIDIA-SMI CUDA Version: $NVIDIA_SMI_VERSION"

version_ge() { printf '%s\n%s' "$2" "$1" | sort -C -V; }

if version_ge $NVCC_VERSION $DESIRED_CUDA_VERSION; then
    echo "NVCC version is correct."
else
    echo "NVCC version is not $DESIRED_CUDA_VERSION."
    exit 1
fi

if version_ge $NVIDIA_SMI_VERSION $DESIRED_CUDA_VERSION; then
    echo "NVIDIA-SMI version is correct."
else
    echo "NVIDIA-SMI version is not $DESIRED_CUDA_VERSION."
    exit 1
fi

export WANDB__SERVICE_WAIT=300

python benchmark.py --config_file=src/benchmark/tuned_config/higgs/mlp_higgs_tuned.json --num_runs=20

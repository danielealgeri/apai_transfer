#!/bin/bash
# run_cuda.sh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-00:05:00
#SBATCH --output slurm-%j.out
nvcc k-means_COMMENTED.cu -o cuda-program
echo "=== CUDA program starts ==="
srun ./cuda-program 5 iris.txt out-cuda.txt
echo "=== End of Job ==="

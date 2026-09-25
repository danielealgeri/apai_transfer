#!/bin/bash
# run-serial.sh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=0-00:05:00
#SBATCH --output slurm-%j.out
gcc -std=c99 -Wall -Wpedantic k-means.c -o serial-program
export OMP_NUM_THREADS=$SL
srun ./serial-program 5 iris.txt out-serial.txt
echo "== End of Job =="
